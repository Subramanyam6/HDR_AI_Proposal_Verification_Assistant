import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { X } from 'lucide-react';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function AboutModal({ isOpen, onClose }: AboutModalProps) {
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      onClick={onClose}
    >
      <Card
        className="max-w-2xl w-full mx-4 max-h-[80vh] overflow-hidden border-2 border-black shadow-brutal bg-background"
        onClick={(e) => e.stopPropagation()}
      >
        <CardHeader className="flex flex-row items-center justify-between border-b-2 border-black bg-background">
          <CardTitle className="text-2xl font-bold">Welcome to HDR Proposal Verification Assistant</CardTitle>
          <button
            onClick={onClose}
            className="p-1 bg-destructive text-destructive-foreground hover:bg-red-600 border-2 border-black shadow-brutal rounded transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </CardHeader>
        <CardContent className="p-6 overflow-y-auto bg-background" style={{ maxHeight: 'calc(80vh - 80px)' }}>
          <div className="space-y-4 text-sm">
            <div>
              <h3 className="font-bold text-base mb-2">Overview</h3>
              <p className="text-muted-foreground break-words">
                This application provides automated compliance verification for HDR proposal documents using 
                machine learning models and rule-based checks. It identifies potential issues and suggests 
                AI-powered fixes to ensure proposals meet HDR standards.
              </p>
            </div>
            
            <div>
              <h3 className="font-bold text-base mb-2">Features</h3>
              <ul className="list-disc list-inside space-y-1 text-muted-foreground break-words">
                <li>Multi-label classification using Naive Bayes, TF-IDF + Logistic Regression, and DistilBERT</li>
                <li>Rule-based date inconsistency detection</li>
                <li>Crosswalk error identification</li>
                <li>Banned phrases detection</li>
                <li>Name inconsistency checking</li>
                <li>AI-powered fix suggestions using GPT models</li>
              </ul>
            </div>

            <div>
              <h3 className="font-bold text-base mb-2">How It Works</h3>
              <ol className="list-decimal list-inside space-y-1 text-muted-foreground break-words">
                <li>Upload a PDF proposal or paste proposal text</li>
                <li>Run verification to check compliance</li>
                <li>Review rule-based and ML-based check results</li>
                <li>Get AI-powered suggestions for fixing identified issues</li>
              </ol>
            </div>

            <div>
              <h3 className="font-bold text-base mb-2">Technology Stack</h3>
              <p className="text-muted-foreground break-words">
                Built with React, FastAPI, Python, TypeScript, and powered by DistilBERT, 
                TF-IDF + Logistic Regression, Naive Bayes, and GPT models.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

