# 🧪 HDR Proposal Test Cases

## Test Case 1: Clean Proposal (No Issues)
```
HDR Engineering Services Proposal
Project: Downtown Infrastructure Improvement
Client: City of Springfield
Project Manager: John Smith
Date: March 15, 2024

We propose to deliver comprehensive engineering services for the downtown infrastructure improvement project. Our team has extensive experience in municipal engineering projects and will ensure all deliverables meet the specified requirements.

Project Scope:
- Traffic analysis and design
- Utility coordination
- Environmental compliance
- Construction oversight

Timeline: 12 months from contract execution
Budget: $2.5M as specified in RFP
```

## Test Case 2: Banned Phrases Detected
```
HDR Engineering Services Proposal
Project: Downtown Infrastructure Improvement
Client: City of Springfield
Project Manager: John Smith
Date: March 15, 2024

We guarantee success on this project. Our team promises to deliver exceptional results that will exceed all expectations. We assure you that this project will be completed flawlessly.

We commit to providing the best possible outcome and pledge our reputation on delivering superior performance.
```

## Test Case 3: Name Inconsistency
```
HDR Engineering Services Proposal
Project: Downtown Infrastructure Improvement
Client: City of Springfield
Project Manager: John Smith
Date: March 15, 2024

Project Team:
- Lead Engineer: John Smith
- Contact Person: J. Smith
- Technical Lead: Johnny Smith
- Project Coordinator: John A. Smith

All communications should be directed to John Smith.
```

## Test Case 4: Date Inconsistency
```
HDR Engineering Services Proposal
Project: Downtown Infrastructure Improvement
Client: City of Springfield
Project Manager: John Smith
Date: March 15, 2024

Project Timeline:
- Phase 1: April 2024
- Phase 2: May 2024
- Phase 3: June 2024
- Completion: July 2024

Note: All dates are based on 2024 calendar year.
```

## Test Case 5: Crosswalk Error (Requirements Mismatch)
```
HDR Engineering Services Proposal
Project: Downtown Infrastructure Improvement
Client: City of Springfield
Project Manager: John Smith
Date: March 15, 2024

RFP Requirements:
- Minimum 5 years experience
- Licensed Professional Engineer
- Previous municipal work

Our Proposal:
- 3 years experience (as requested)
- Certified Engineer (not licensed)
- Commercial work only (no municipal)
```

## Test Case 6: Multiple Issues Combined
```
HDR Engineering Services Proposal
Project: Downtown Infrastructure Improvement
Client: City of Springfield
Project Manager: John Smith
Date: March 15, 2024

We guarantee this project will succeed beyond expectations. Our team promises exceptional results.

Project Team:
- Lead: John Smith
- Contact: J. Smith
- Manager: Johnny Smith

Timeline:
- Start: April 2024
- Finish: June 2025
- Delivery: July 2024

We assure you of our commitment to excellence.
```

## Test Case 7: Edge Case - Very Short Text
```
HDR Proposal. John Smith. 2024.
```

## Test Case 8: Edge Case - Very Long Text
```
HDR Engineering Services Proposal for Downtown Infrastructure Improvement Project

Client: City of Springfield
Project Manager: John Smith
Date: March 15, 2024

Executive Summary:
We propose to deliver comprehensive engineering services for the downtown infrastructure improvement project. Our team has extensive experience in municipal engineering projects and will ensure all deliverables meet the specified requirements.

Project Scope:
The project involves comprehensive infrastructure improvements including traffic analysis, utility coordination, environmental compliance, and construction oversight. We will provide detailed engineering designs, permit assistance, and construction management services.

Technical Approach:
Our methodology includes detailed site analysis, traffic impact studies, utility coordination, environmental assessments, and construction oversight. We will utilize state-of-the-art software and proven engineering practices.

Project Team:
Our team consists of licensed professional engineers with extensive municipal experience. Each team member has over 10 years of relevant experience and appropriate certifications.

Timeline and Budget:
The project will be completed within 12 months of contract execution. The total budget is $2.5M as specified in the RFP. We will provide monthly progress reports and maintain open communication throughout the project.

Quality Assurance:
We maintain rigorous quality control procedures and will ensure all deliverables meet or exceed the specified requirements. Our team is committed to delivering exceptional results.

Conclusion:
We are confident in our ability to deliver this project successfully. Our team's experience and commitment to excellence make us the ideal choice for this important infrastructure improvement project.
```

## Expected Results:

**Test Case 1:** ✅ All checks should pass
**Test Case 2:** ❌ Banned phrases detected
**Test Case 3:** ❌ Name inconsistency detected  
**Test Case 4:** ❌ Date inconsistency detected
**Test Case 5:** ❌ Crosswalk error detected
**Test Case 6:** ❌ Multiple issues detected
**Test Case 7:** Edge case testing
**Test Case 8:** Edge case testing

---

## How to Use:

1. Copy any test case text
2. Paste into the text input field
3. Click "Verify Proposal"
4. Check results in all three tabs
5. Compare DistilBERT vs TF-IDF performance

