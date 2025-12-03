from google.adk.agents import Agent, ParallelAgent, SequentialAgent, LoopAgent
from google.adk.apps.app import App, ResumabilityConfig
from google.adk.models.google_llm import Gemini
from google.adk.tools import AgentTool, google_search, load_artifacts
from google.adk.plugins.save_files_as_artifacts_plugin import SaveFilesAsArtifactsPlugin
from google.adk.tools.tool_context import ToolContext
from google.genai import types
from pathlib import Path

def resume_gap_asker_tool(tool_context: ToolContext) -> dict:
    """
    Tool to ask user for missing information based on critique.
    Allows user to skip and proceed without providing details.
    Args:
        tool_context (ToolContext): The context of the tool, including state and confirmation handling.
    Returns:
        dict: A dictionary indicating the status and user input.
    """
    critique = tool_context.state.get("critique", "")
    
    tool_confirmation = tool_context.tool_confirmation
    
    # If no confirmation yet, request it and return waiting status
    if not tool_confirmation:
        hint = (
            f"To improve your resume, please provide: {critique}\n\n"
            "You can also type 'SKIP' to proceed with current resume."
        )
        tool_context.request_confirmation(hint=hint, payload={"user_response": "your input here"})

        return {
            "status": "awaiting_user_input",
            "message": "Waiting for user response..."
        }
    
    try:
        # Confirmation is now available, extract payload
        user_response = tool_confirmation.payload["user_response"]
        
        #initialized the user_input in state if not present
        user_input = tool_context.state.get("user_input", "")
        user_input = user_response
        tool_context.state["user_input"] = user_input

        # Check if user wants to skip
        if user_response.upper() == "SKIP":
            return {
                "status": "skipped",
                "user_input": "User chose to proceed without additional information",
                "message": "Proceeding with current resume..."
            }
        
        return {
            "status": "info_collected",
            "user_input": user_response
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }

def get_latex_template_tool() -> str:
    """
    Retrieves the LaTeX resume template from file.
    Returns:
        str: The raw LaTeX template string with placeholders.
    """
    try:
        template_path = Path(__file__).parent / "resume.tex"
        
        with open(template_path, 'r', encoding='utf-8') as f:
            latex_content = f.read()
        
        return latex_content
    
    except FileNotFoundError:
        raise FileNotFoundError(f"LaTeX template not found at: {template_path}")
    except Exception as e:
        raise Exception(f"Error reading LaTeX template: {str(e)}")

async def save_generated_resume_latex(tool_context: ToolContext, latex_content: str):
    """Saves generated latex content as an artifact."""

    # Convert the string content back into bytes for artifact handling
    latex_bytes = latex_content.encode('utf-8')

    artifact_part = types.Part(
        inline_data=types.Blob(
            mime_type="text/plain",
            data=latex_bytes
        )
    )
    filename = "generated_resume_latex.tex"

    try:
        version = await tool_context.save_artifact(filename=filename, artifact=artifact_part)
        return {
        "status": "success",
        "message": f"File '{filename}' (version {version}) has been created and is now available for download.",
        # The ADK UI will automatically intercept this response and provide a download link.
    }
    except ValueError as e:
        print(f"Error saving LaTeX artifact: {e}. Is ArtifactService configured in Runner?")
    except Exception as e:
        print(f"An unexpected error occurred during LaTeX artifact save: {e}")

# --- Configuration ---
retry_config = types.HttpRetryOptions(
    attempts=3,
    initial_delay=1,
    http_status_codes=[429, 500, 503],
)


model_standard = Gemini(model="gemini-2.5-flash", retry_options=retry_config)
model_lite = Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config)

# 1. RESUME PARSER 
resume_parser_agent = Agent(
    name="ResumeParserAgent",
    model=model_standard, 
    instruction="""You are a resume parsing specialist. Extract the user’s resume into a clean JSON structure without changing facts.

    Inputs: Resume content.

    Output: A JSON with the following fields:
    - full_name
    - contact_info: {email, phone, linkedin, github(if relevant), portfolio(if relevant)}
    - education[]: {institution, degree, {city, state/country}, start_date(YYYY-MM), end_date(YYYY-MM), gpa(or percentage)}
    - experience[]: {company, title, location, start_date, end_date, description}
    - projects[]: {name, description, technologies{}, link(must if relevant), impact_bullets[]}
    - skills[]: {hard_skills, soft_skills} list of skills (divide into categories of object{} if possible)
    - certifications[]: {name, issuer, date}list of certifications
    - achievements[]: {title, description{}}list of achievements
    - languages[]: list of languages with proficiency levels
    Output ONLY the structured JSON to the state with the key 'resume_structured'.
""",
    output_key="resume_structured" 
)

# 2. JOB DESCRIPTION PARSER
job_description_parser_agent = Agent(
    name="JobDescriptionParserAgent",
    model=model_standard, 
    instruction="""You are a job description analysis expert. Convert the given job description into a requirements map.

	Inputs: Look for the job description text in the user's chat first; if not found, call the `load_artifacts` tool to retrieve an uploaded job description file.
    Description file from calling `load_artifacts` tool.
	Output: JSON with:
	role_title, seniority, location,
	must_have_skills[],
	nice_to_have_skills[],
	responsibilities[],
	keywords[] (phrases important for ATS),
	screen_out_criteria[] (e.g. ‘requires 5+ years’).
	Do not copy long sentences; keep items short and specific.
	Output ONLY the structured JSON to the session context with the key 'jd_structured'.
    """,
    output_key="jd_structured",
    tools=[load_artifacts],
)

# 3. COMPANY RESEARCHER 
company_researcher_agent = Agent(
    name="ResearcherAgent",
    model=model_standard,
    instruction="""You are a Company and Job Profile Analyst.

    Inputs:
    - `company_name`: The name of the company to research.

    Your Goal:
    Create a detailed "Job Profile" by researching the company.

    Steps:
    1.  Company Research:
        - Use the `google_search` tool to find information about the company's:
            - Mission, values, and culture (e.g., search for "company_name company culture", "company_name mission and values").
            - Recent news or significant events (e.g., search for "company_name recent news").
            - Products, services, and core business.
        - Synthesize these findings into a "company_overview" section.

    Output:
    - A JSON object assigned to the `job_profile` key with the following structure:
    {{
        "company_overview": {{
            "mission": "...",
            "values": ["...", "..."],
            "culture_summary": "...",
            "recent_news_summary": "..."
        }},
        "role_alignment": {{
            "top_3_priorities": ["...", "...", "..."],
            "critical_skills": ["...", "..."],
            "ideal_candidate_profile": "..."
        }}
    }}
    - The "ideal_candidate_profile" should be a 2-3 sentence summary of the perfect candidate, blending skills with company culture.
    - Keep all summaries concise and focused on what a job applicant needs to know.
    """,
    tools=[google_search],
    output_key="job_profile"
)

# 4. DATA PARSER
data_parser = ParallelAgent(
    name="DataParserAgent",
    sub_agents=[resume_parser_agent, job_description_parser_agent, company_researcher_agent], 
)

#5. INITIAL RESUME OPTIMIZATION
initial_resume_optimization_agent = Agent(
    name="InitialResumeOptimizationAgent",
    model=model_standard,
    instruction="""You are a concise resume optimizer. Your goal is to generate an initial optimized resume draft that will fit on a single page.

    Inputs: {resume_structured} (your resume data), {jd_structured} (job requirements), and {job_profile} (company insights).
    Output: An optimized JSON resume, highlighting alignment with the job description and company culture.
    
    Important:
    - The output resume must remain factual; do not invent new experiences or skills or projects.
    - Only rephrase and reorganize existing information to better match the job description.
    - Make sure the json structure remains consistent with the original resume format.

    Content Rules:
    -No Summary or Objective sections.
    - Select the AT MOST 3 relevant projects and 3 relevant experiences from the user's resume based on the job description.
    - each description of project or experience should concise, short and NOT MORE THAN 2 sentences.
    - The sentence should be CONCISE, achievement-oriented, quantifying impact where possible.
    - Words MUST be between 350-450 words to ensure ATS compatibility and to FIT IN A SINGLE PAGE.

    Output ONLY the optimized resume JSON to the session context with the key 'optimized_resume'.
    """,
    output_key="optimized_resume",
)

#6. CRITIQUE AGENT
critique_agent = Agent(
    name="CritiqueAgent",
    model=model_lite,
    instruction="""You are a constructive resume critique and ATS compatibility reviewer.
    Inputs:
        - Resume: {optimized_resume}
        - Job Description: {jd_structured}
        - Company Insights: {job_profile}

    Tasks:
        1. Evaluate the resume for clarity, relevance, structure, and ATS-friendliness.
        2. Verify alignment with the job description and company profile.
        3. If the resume is well-written and fully compatible, respond ONLY with the exact phrase: "APPROVED"
        4. Otherwise, provide concise, actionable suggestions for improvement (e.g., add measurable results, include missing required skills, clarify dates/roles).

    Output only the approval phrase or the suggested improvements as plain text.""",
    output_key="critique",
)

#7. RESUME GAP ASKER
resume_gap_asker = Agent(
    name="ResumeGapAsker",
    model=model_lite,
    instruction="""You are helpful questioniare that asks user about resume and job description gap.
    You have {critique}, and
    Your task is to Call `resume_gap_asker_tool` tool to get the missing information from the user.
    And always return the tools output. 
    """,
    tools=[resume_gap_asker_tool],
    output_key="user_input",
)

#8. RESUME GAP FILLER
resume_gap_filler_agent = Agent(
    name="ResumeGapFillerAgent",
    model=model_standard,
    instruction="""You are a Resume refiner. You have a resume draft and a critique.
    Inputs:
    Resume Draft: {optimized_resume}
    Critique: {critique}
    IMPORTANT:
    - Maintain factual accuracy; DO NOT invent new experiences, skills, or projects.
    Your task is to analyze the critique:
    - IF the critique is EXACTLY "APPROVED", set the status to 'approved' and output the optimized_resume as-is.
    - OTHERWISE, rewrite the resume draft to fully incorporate the feedback from the critique.

    Content Rules:
    -No Summary or Objective sections.
    - Select the AT MOST 3 relevant projects and 3 relevant experiences from the user's resume based on the job description.
    - each description of project or experience should concise, short and NOT MORE THAN 2 sentences.
    - The sentence should be CONCISE, achievement-oriented, quantifying impact where possible.
    - Words MUST be between 350-450 words to ensure ATS compatibility and to FIT IN A SINGLE PAGE.
    
    Output the Resume JSON in ALL cases as the agent response.
""",
    output_key="optimized_resume",
)

# 9. RESUME LATEX AGENT
resume_latex_agent = Agent(
    name="ResumeLatexAgent",
    model=model_standard,
    instruction="""You are a specialist LaTeX resume generator. Your only job is to populate a LaTeX template with the provided JSON data.

    Inputs:
            - Resume Data: {optimized_resume}
            - LaTeX Template: Call the `get_latex_template_tool` to retrieve the raw LaTeX_TEMPLATE"
    
    Rules:
        1.  You MUST NOT edit, alter, or remove any of the existing LaTeX syntax from the template. Your only task is to replace the placeholder content.
        2.  To ensure the resume fits on a single page, you may remove irrelevant projects or experiences. However, you MUST NOT shorten the descriptions of the items you decide to keep.
        3.  Do not add extra curly braces. The template commands are correct. (Bad: resumeItem{{}{}}, Good: resumeItem{} {}).
        4.  If there is Escape special characters (like %, $, &) present in the content, Use single backslash before that character(like \%, \$, \&).
        5. DO NOT use any packages or commands not already present in the LaTeX template.
        6. DO NOT write any \end commands or document closure commands; the template already includes them.
        7. Use only commands with single \ not double.
        8. Recheck for latex syntax correctness remove errors if any present in the content.

    Tasks:
        1.  Fill the template using the data from the 'optimized_resume' JSON.
        2.  For each experience and project, iterate through the 'impact_bullets' array and write a 'resumeItem' for each bullet.
        3. Ensure all sections (Education, Experience, Projects, Skills, Achievements, Languages) are properly populated at the right section. Remmove any section that has no data.
        4. If any information is not relevant to the section of Latex template, find the best fitting section to include it.
        5. Ensure all the sections have relevant data in them.
        8. OUTPUT only the final, 'filled_LaTeX_TEMPLATE'.
    """,
    tools=[load_artifacts, get_latex_template_tool],
    output_key="filled_LaTeX_TEMPLATE",
)

# 10. RESUME SEQUENCE
resume_sequence = SequentialAgent(
    name="ResumeSequence",
    description="Executes the initial resume optimization process. Requires 'resume', 'job_description', and 'company_name'.",
    sub_agents=[data_parser, initial_resume_optimization_agent, critique_agent, resume_gap_asker, resume_gap_filler_agent, resume_latex_agent],
)

# 0. ROOT AGENT
root_agent = Agent(
    name="RootAgent",
    model=model_standard,
    instruction="""You are a helpful agent.
    RULES:
    1. If the user says "Hi" or asks for help, explain that you need:
        - Their Resume (text or file)
        - The Job Description (text or file)
        - The Company Name
    2. If user upload any file call `load_artifact` tool and get the content.
    3. DO NOT call the `ResumeSequence` tool until you have ALL three pieces of information.
    4. Once you have ALL three pieces, call `load_artifacts` tool to retrieve any uploaded files.
    5. Then call the `ResumeSequence` tool.
    6. After the sequence completes you will get 'filled_LaTeX_TEMPLATE', call the `save_generated_resume_latex` tool.
    7. Call the `load_artifacts` tool again to verify the save.
    8. Confirm to the user that their resume has been successfully generated and saved.
    9. Tell the user to download the resume and tell them how they can edit there resume or convert it into pdf using online latex editors.
    """,
    tools=[AgentTool(resume_sequence), load_artifacts, save_generated_resume_latex],
)

app = App(
    name="my_agent",
    root_agent=root_agent,
    plugins=[SaveFilesAsArtifactsPlugin()] # <--- This handles the upload logic
)