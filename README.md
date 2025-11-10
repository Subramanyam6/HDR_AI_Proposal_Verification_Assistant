# HDR AI Proposal Verification Assistant

An AI-powered tool for automated compliance verification of HDR proposal documents using machine learning models and rule-based checks.

## Features

- **Multi-label Classification**: Uses Naive Bayes, TF-IDF + Logistic Regression, and DistilBERT models
- **Rule-based Checks**: Date inconsistency detection
- **AI-powered Fix Suggestions**: GPT-5/GPT-4o powered recommendations
- **PDF & Text Input**: Upload PDFs or paste proposal text directly

## Models Used

- **DistilBERT**: Primary semantic check for crosswalk errors, banned phrases, and name inconsistencies
- **TF-IDF + Logistic Regression**: Literal phrase pattern matching
- **Naive Bayes**: Fast baseline sanity check

## Technology Stack

- Frontend: React, TypeScript, Tailwind CSS, Vite
- Backend: FastAPI, Python, PyTorch, Transformers
- ML Models: DistilBERT, scikit-learn
- AI Suggestions: OpenAI GPT-5/GPT-4o

## Usage

1. Upload a PDF proposal or paste proposal text
2. Click "Run Verification" to check compliance
3. Review rule-based and ML-based check results
4. Get AI-powered suggestions for fixing identified issues

## License

Apache 2.0 License - Copyright 2025 Bala Subramanyam Duggirala, HDR Inc.
