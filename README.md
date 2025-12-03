# 🎯 Resume Optimizer AI Agent

An intelligent AI-powered agent that transforms your resume by analyzing it against job descriptions and generating an optimized, editable LaTeX version tailored to specific job requirements.

## 📖 Overview

This Resume Optimizer AI Agent takes your existing resume (in PDF or text format) along with a job description and intelligently restructures and optimizes it to create a professionally formatted, ATS-friendly resume in editable LaTeX format. The AI agent analyzes the job requirements and tailors your resume content to maximize your chances of landing an interview.

## ✨ Key Features

- **📄 Multiple Input Formats** - Upload resumes in PDF or plain text format
- **🎯 Job Description Analysis** - Intelligent matching between your experience and job requirements
- **🤖 AI-Powered Optimization** - Uses advanced AI to restructure and enhance resume content
- **📝 LaTeX Output** - Generates professionally formatted, editable LaTeX resumes
- **🔑 Keyword Optimization** - Automatically incorporates relevant keywords from job descriptions
- **⚡ ATS-Friendly** - Ensures compatibility with Applicant Tracking Systems
- **✏️ Fully Editable** - Output in LaTeX format for easy customization
- **🎨 Professional Templates** - Clean, modern resume formatting

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- LaTeX distribution (for compiling LaTeX files)
- Gemini API key or compatible AI service

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Tushar-Sahu7/Resume-Optimizer-AI-Agent.git
cd Resume-Optimizer-AI-Agent
```

2. **Create and activate virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install required dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
# Create a .env file in the root directory
cp .env.example .env

# Add your API keys to .env
GEMINI_API_KEY=your_api_key_here
GOOGLE_GENAI_USE_VERTEXAI=0
```

5. **Run the application**
```bash
adk run
# or
adk web
```

## 💻 Usage

### Basic Workflow

1. **Upload Your Resume**: Provide your current resume in PDF or text format
2. **Add Job Description**: Paste or upload the target job description
3. **Generate Optimized Resume**: Let the AI agent analyze and optimize
4. **Download LaTeX File**: Receive your professionally formatted resume in LaTeX
5. **Compile and Customize**: Edit the LaTeX file and compile to PD


## 📊 How It Works

1. **Resume Parsing**: Extracts text and structure from your uploaded resume
2. **Job Analysis**: Uses NLP to identify key requirements, skills, and keywords from the job description
3. **Content Matching**: AI agent maps your experience to job requirements
4. **Optimization**: Restructures content, enhances descriptions, and incorporates relevant keywords
5. **LaTeX Generation**: Creates a professionally formatted LaTeX document

## 🎨 LaTeX Output Features

- Clean, professional formatting
- ATS-compatible structure
- Customizable sections (Education, Experience, Skills, Projects)
- Consistent typography and spacing
- Easy to edit and recompile
- Multiple template options


## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## 📧 Contact

Tushar Sahu - [@Tushar-Sahu7](https://github.com/Tushar-Sahu7)

Project Link: [https://github.com/Tushar-Sahu7/Resume-Optimizer-AI-Agent](https://github.com/Tushar-Sahu7/Resume-Optimizer-AI-Agent)

## 🐛 Issues & Support

If you encounter any issues or have questions:
- Check the [Issues](https://github.com/Tushar-Sahu7/Resume-Optimizer-AI-Agent/issues) page
- Create a new issue with detailed information
- Join discussions in the project community

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐
