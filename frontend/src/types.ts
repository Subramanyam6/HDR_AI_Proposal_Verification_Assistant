export interface ModelResult {
  "Crosswalk Error": boolean;
  "Banned Phrases": boolean;
  "Name Inconsistency": boolean;
}

export interface RuleResult {
  date_inconsistency: boolean;
}

export interface VerificationResponse {
  ml_results: {
    transformer: ModelResult;
    tfidf: ModelResult;
    naive_bayes: ModelResult;
  };
  rule_results: RuleResult;
  suggestions: string | null;
}

export interface SampleLibrary {
  [key: string]: string;
}
