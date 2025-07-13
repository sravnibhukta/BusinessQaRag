"""
Demo responses for RAG system when OpenAI API is not available
"""

DEMO_RESPONSES = {
    "What is the company's vacation policy?": {
        "answer": "According to our employee handbook, ACME Corporation offers 15 days of paid vacation for employees with 0-2 years of service, 20 days for 3-7 years, and 25 days for 8+ years. Vacation time must be requested at least 2 weeks in advance and approved by your manager. Unused vacation days can be carried over up to 5 days to the next year.",
        "sources": [
            {
                "content": "ACME Corporation Vacation Policy: Full-time employees are eligible for paid vacation time based on their years of service. 0-2 years: 15 days, 3-7 years: 20 days, 8+ years: 25 days. All vacation requests must be submitted at least 2 weeks in advance...",
                "metadata": {"source": "employee_handbook.txt"},
                "score": 0.92
            },
            {
                "content": "Vacation Carryover Policy: Employees may carry over up to 5 unused vacation days to the following calendar year. Any additional unused days will be forfeited...",
                "metadata": {"source": "business_policy.txt"},
                "score": 0.85
            }
        ]
    },
    "What are the dress code requirements?": {
        "answer": "ACME Corporation maintains a business casual dress code. Employees should dress professionally while maintaining comfort. Acceptable attire includes collared shirts, blouses, slacks, skirts, and closed-toe shoes. Avoid shorts, flip-flops, tank tops, and overly casual clothing. On Fridays, casual dress is permitted including jeans and sneakers.",
        "sources": [
            {
                "content": "Dress Code Policy: ACME Corporation expects all employees to maintain a professional appearance. Business casual attire is required Monday through Thursday. Acceptable items include collared shirts, blouses, dress pants, skirts...",
                "metadata": {"source": "employee_handbook.txt"},
                "score": 0.89
            }
        ]
    },
    "What health benefits does the company offer?": {
        "answer": "ACME Corporation provides comprehensive health benefits including medical, dental, and vision insurance. The company covers 80% of premiums for employees and 60% for dependents. Additional benefits include health savings account (HSA) options, mental health support, and an annual wellness program with fitness reimbursements up to $500.",
        "sources": [
            {
                "content": "Health Benefits Overview: ACME Corporation offers medical, dental, and vision insurance through Blue Cross Blue Shield. Employee premium contribution is 20% with company covering 80%. Dependent coverage available at 60% company contribution...",
                "metadata": {"source": "employee_handbook.txt"},
                "score": 0.94
            },
            {
                "content": "Wellness Program: Annual fitness reimbursement up to $500 for gym memberships, fitness classes, or health-related activities. Mental health support through Employee Assistance Program (EAP) available 24/7...",
                "metadata": {"source": "business_policy.txt"},
                "score": 0.87
            }
        ]
    },
    "How do I report a workplace injury?": {
        "answer": "In case of a workplace injury, immediately notify your supervisor and seek medical attention if needed. Report the incident to HR within 24 hours using the incident report form. Contact the workers' compensation hotline at 1-800-WORKCOMP. All workplace injuries must be documented regardless of severity to ensure proper tracking and support.",
        "sources": [
            {
                "content": "Workplace Injury Reporting: All workplace injuries must be reported immediately to supervisors and HR within 24 hours. Use the official incident report form available on the company intranet. Contact workers' compensation hotline...",
                "metadata": {"source": "business_policy.txt"},
                "score": 0.91
            }
        ]
    },
    "What is the process for expense reports?": {
        "answer": "Submit expense reports monthly through the company portal by the 5th of each month. Include original receipts for all expenses over $25. Categories include travel, meals, supplies, and professional development. Reports must be approved by your manager before processing. Reimbursement typically takes 5-7 business days after approval.",
        "sources": [
            {
                "content": "Expense Report Process: Monthly submission required by 5th of each month. Original receipts needed for expenses over $25. Submit through company portal with manager approval required. Processing time 5-7 business days...",
                "metadata": {"source": "business_policy.txt"},
                "score": 0.88
            }
        ]
    },
    "How can I apply for a job at ACME Corporation?": {
        "answer": "To apply for positions at ACME Corporation, visit our careers page at acme.com/careers. Submit your resume and cover letter through our online application system. We review applications on a rolling basis and will contact qualified candidates within 2 weeks. Current employees can refer candidates through our internal referral program.",
        "sources": [
            {
                "content": "Job Application Process: Visit acme.com/careers to view current openings. Submit applications through online portal. Application review process takes up to 2 weeks. Employee referral program available for internal recommendations...",
                "metadata": {"source": "company_faq.txt"},
                "score": 0.86
            }
        ]
    }
}

def get_demo_response(query: str) -> dict:
    """
    Get a demo response for a given query
    
    Args:
        query: User query
        
    Returns:
        Dictionary with answer and sources
    """
    # Check for exact match first
    if query in DEMO_RESPONSES:
        return DEMO_RESPONSES[query]
    
    # Check for partial matches
    for demo_query, response in DEMO_RESPONSES.items():
        if any(word.lower() in query.lower() for word in demo_query.split() if len(word) > 3):
            return response
    
    # Default response
    return {
        "answer": "This is a demo version of the RAG system. The response would normally be generated using OpenAI's API based on your business documents. To see full functionality, please add credits to your OpenAI account and provide a working API key.",
        "sources": [
            {
                "content": "Demo mode active - this would normally search through your uploaded business documents to find relevant information and generate contextual responses.",
                "metadata": {"source": "demo_system.txt"},
                "score": 0.95
            }
        ]
    }