import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.1-8b-instant")

    def extract_jobs(self, cleaned_text):
        prompt_extract = ChatPromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE
            {page_data}

            ### TASK
            Extract all job postings.

            ### OUTPUT FORMAT (STRICT)
            Return a single valid JSON object with these keys:
            -"role"
            - "experience"
            - "skills"
            - "description"

            Do not wrap it in an array.
            Do not add any text outside the JSON.

            ### RULES
            - Keep the **original phrasing of skills as much as possible**.
            - Group related terms together logically (e.g., `"AWS - EC2, ECS, API Gateway"`, not `"AWS"`, `"EC2"`, `"ECS"`, ... separately).
            - If a value is missing/unclear, use an empty string "" (or [] for "skills").
            - Keep JSON valid: no trailing commas, no comments, no code fences.
            - No explanations or text outside the JSON.

            ### VALID JSON (NO PREAMBLE)
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = ChatPromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
        
             ### INSTRUCTION:
            You are Siddharth, a business development executive at XYZ. XYZ is an AI & Software Consulting company dedicated to facilitating
            the seamless integration of business processes through automated tools. 
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability, 
            process optimization, cost reduction, and heightened overall efficiency. 

            Your job is to write a cold email to the client regarding the job mentioned above describing the capability of XYZ 
            in fulfilling their needs.

            You are also given a list of portfolio links: {link_list}
            Use these links in the email naturally, for example:
            - “Here are a few relevant projects: [link]”
            - “You can explore a similar solution here: [link]”
            DO NOT make up any client names or project details. Just use the links as they are.

            Remember:
            - Do not include placeholders like [Client Name].
            - Keep the tone professional and warm.
            - End with a clear call to action.

            ### EMAIL (NO PREAMBLE):

            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))