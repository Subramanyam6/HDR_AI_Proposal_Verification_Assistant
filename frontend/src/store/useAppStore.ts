import { create } from 'zustand';
import axios from 'axios';
import type { VerificationResponse } from '../types';

interface AppState {
  inputText: string;
  pdfFile: File | null;
  selectedSample: string | null;
  selectedPdfSample: string | null;
  samples: Record<string, string>;
  results: VerificationResponse | null;
  isLoading: boolean;
  isSampleLoading: boolean;
  error: string | null;

  setInputText: (text: string) => void;
  setPdfFile: (file: File | null) => void;
  setSelectedSample: (sample: string | null) => void;
  setSelectedPdfSample: (sample: string | null) => void;
  loadSamples: () => Promise<void>;
  runVerification: () => Promise<void>;
  regenerateSuggestions: () => Promise<void>;
  clearAll: () => void;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const useAppStore = create<AppState>((set, get) => ({
  inputText: '',
  pdfFile: null,
  selectedSample: null,
  selectedPdfSample: null,
  samples: {},
  results: null,
  isLoading: false,
  isSampleLoading: false,
  error: null,

  setInputText: (text) => set({ inputText: text, pdfFile: null, selectedSample: null, selectedPdfSample: null }),

  setPdfFile: (file) => set({ pdfFile: file, inputText: '', selectedSample: null, selectedPdfSample: null }),

  loadSamples: async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/samples`);
      set({ samples: response.data });
    } catch (error) {
      console.error('Failed to load samples:', error);
    }
  },

  setSelectedSample: (sampleKey: string | null) => {
    if (!sampleKey || sampleKey === 'Select a text sample to upload...') {
      set({ selectedSample: null, inputText: '', pdfFile: null, selectedPdfSample: null, isSampleLoading: false });
      return;
    }

    const { samples } = get();
    if (samples[sampleKey]) {
      set({ isSampleLoading: true, pdfFile: null, selectedPdfSample: null });

      // 1-second delay before loading sample text
      setTimeout(() => {
        set({
          inputText: samples[sampleKey],
          selectedSample: sampleKey,
          isSampleLoading: false
        });
      }, 1000);
    } else {
      console.warn(`Sample "${sampleKey}" not found in loaded samples`);
      set({ error: `Sample "${sampleKey}" not found`, isSampleLoading: false });
    }
  },

  setSelectedPdfSample: async (sampleKey: string | null) => {
    if (!sampleKey || sampleKey === 'Select a PDF sample to upload...') {
      set({ selectedPdfSample: null, pdfFile: null, inputText: '', selectedSample: null });
      return;
    }

    try {
      set({ isSampleLoading: true, inputText: '', selectedSample: null });
      
      // Download PDF from backend
      const response = await axios.get(`${API_BASE_URL}/samples/pdf/${encodeURIComponent(sampleKey)}`, {
        responseType: 'blob'
      });
      
      // Create File object from blob
      const blob = response.data;
      const pdfFile = new File([blob], `${sampleKey}.pdf`, { type: 'application/pdf' });
      
      set({
        pdfFile,
        selectedPdfSample: sampleKey,
        isSampleLoading: false
      });
    } catch (error) {
      console.error('Failed to load PDF sample:', error);
      set({ 
        error: `Failed to load PDF sample: ${sampleKey}`,
        isSampleLoading: false,
        selectedPdfSample: null
      });
    }
  },

  runVerification: async () => {
    const { inputText, pdfFile } = get();

    if (!inputText && !pdfFile) {
      set({ error: 'Please provide input text or upload a PDF' });
      return;
    }

    set({ isLoading: true, error: null, results: null });

    try {
      const formData = new FormData();

      if (pdfFile) {
        formData.append('file', pdfFile);
      } else if (inputText) {
        formData.append('text', inputText);
      }

      console.log('Sending verification request to:', `${API_BASE_URL}/verify`);
      const response = await axios.post<VerificationResponse>(
        `${API_BASE_URL}/verify`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      console.log('Verification response:', response.data);
      // Set results immediately (ML tables show instantly)
      // Suggestions will be null initially if still generating
      set({ results: response.data, isLoading: false });
    } catch (error) {
      console.error('Verification failed:', error);
      const errorMessage = error instanceof Error 
        ? error.message 
        : axios.isAxiosError(error)
        ? error.response?.data?.detail || error.message || 'Verification failed'
        : 'Verification failed';
      set({
        error: errorMessage,
        isLoading: false,
        results: null
      });
    }
  },

  regenerateSuggestions: async () => {
    const { inputText, pdfFile, results } = get();

    if (!inputText && !pdfFile) {
      set({ error: 'Please provide input text or upload a PDF' });
      return;
    }

    if (!results) {
      return;
    }

    try {
      const formData = new FormData();

      if (pdfFile) {
        formData.append('file', pdfFile);
      } else if (inputText) {
        formData.append('text', inputText);
      }

      // Extract failed checks from current results (only from DistilBERT transformer)
      const failedChecks: string[] = [];
      if (results.ml_results) {
        const transformer = results.ml_results.transformer || {};
        
        if (transformer['Crosswalk Error']) {
          failedChecks.push('Crosswalk Error');
        }
        if (transformer['Banned Phrases']) {
          failedChecks.push('Banned Phrases');
        }
        if (transformer['Name Inconsistency']) {
          failedChecks.push('Name Inconsistency');
        }
      }
      
      if (failedChecks.length > 0) {
        formData.append('failed_checks', failedChecks.join(','));
      }

      const response = await axios.post<{ suggestions: string }>(
        `${API_BASE_URL}/suggestions`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      // Update only the suggestions in results
      set({ 
        results: {
          ...results,
          suggestions: response.data.suggestions
        }
      });
    } catch (error) {
      console.error('Failed to regenerate suggestions:', error);
      const errorMessage = error instanceof Error 
        ? error.message 
        : axios.isAxiosError(error)
        ? error.response?.data?.detail || error.message || 'Failed to regenerate suggestions'
        : 'Failed to regenerate suggestions';
      set({ error: errorMessage });
    }
  },

  clearAll: () => set({
    inputText: '',
    pdfFile: null,
    selectedSample: null,
    selectedPdfSample: null,
    results: null,
    error: null,
    isSampleLoading: false,
  }),
}));
