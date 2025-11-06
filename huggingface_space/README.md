---
title: HDR Proposal Verification Assistant
emoji: 🔍
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
---

# 🔍 HDR Proposal Verification Assistant

An AI-powered compliance verification system for HDR proposals, designed to automatically detect common issues and errors.

## Features

- **Automated Compliance Checking**: Detects multiple issue types in one scan
- **Fast Processing**: Results in seconds
- **Clean, Modern UI**: Intuitive interface with detailed results
- **Confidence Scores**: Shows prediction confidence for transparency

## What It Detects

- ✅ **Crosswalk Errors**: Mismatches in project requirements
- ✅ **Banned Phrases**: Prohibited language or terms
- ✅ **Name Inconsistencies**: Name variations throughout the document
- ✅ **Date Inconsistencies**: Conflicting dates in the proposal

## How to Use

1. **Upload**: Select your HDR proposal PDF file
2. **Verify**: Click the "Verify Proposal" button
3. **Review**: Check the results and confidence scores
4. **Preview**: View extracted text (optional)

## Model Details

- **Architecture**: TF-IDF + Logistic Regression (OneVsRest)
- **Features**: 10,000 n-gram features (unigrams + bigrams)
- **Training Data**: 7,200 synthetic HDR proposals
- **Performance**: Micro-F1: 0.410 | Macro-F1: 0.414

## Technical Stack

- **ML Framework**: scikit-learn
- **UI Framework**: Gradio
- **PDF Processing**: PyPDF2
- **Deployment**: Hugging Face Spaces

## Limitations

- Currently supports English text only
- Requires text-extractable PDFs (not scanned images)
- Trained on synthetic data; real-world performance may vary

## Future Enhancements

- [ ] Add DistilBERT/BERT transformer model option
- [ ] Support for scanned PDFs (OCR)
- [ ] Additional verification rules
- [ ] Batch processing
- [ ] Detailed error explanations

## Author

**Subramanyam Duggirala**
GitHub: [@Subramanyam6](https://github.com/Subramanyam6)

## License

MIT License - see the full repository for details.

---

*Built with ❤️ for better proposal compliance*
