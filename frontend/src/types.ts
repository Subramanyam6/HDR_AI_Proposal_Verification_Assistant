export interface ModelResult {
  crosswalk_error: boolean;
  banned_phrases: boolean;
  name_inconsistency: boolean;
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
  suggestions: string;
}

export interface SampleLibrary {
  [key: string]: string;
}
