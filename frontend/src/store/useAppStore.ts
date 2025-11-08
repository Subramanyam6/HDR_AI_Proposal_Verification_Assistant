import { create } from 'zustand';
import axios from 'axios';
import { VerificationResponse } from '../types';

interface AppState {
  inputText: string;
  pdfFile: File | null;
  selectedSample: string | null;
  results: VerificationResponse | null;
  isLoading: boolean;
  error: string | null;

  setInputText: (text: string) => void;
  setPdfFile: (file: File | null) => void;
  setSelectedSample: (sample: string) => void;
  runVerification: () => Promise<void>;
  clearAll: () => void;
}

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

export const useAppStore = create<AppState>((set, get) => ({
  inputText: '',
  pdfFile: null,
  selectedSample: null,
  results: null,
  isLoading: false,
  error: null,

  setInputText: (text) => set({ inputText: text, pdfFile: null }),

  setPdfFile: (file) => set({ pdfFile: file, inputText: '' }),

  setSelectedSample: async (sampleKey) => {
    if (!sampleKey || sampleKey === 'Select a text sample...') {
      set({ selectedSample: null });
      return;
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/samples`);
      const samples = response.data;
      if (samples[sampleKey]) {
        set({ inputText: samples[sampleKey], selectedSample: sampleKey, pdfFile: null });
      }
    } catch (error) {
      console.error('Failed to load sample:', error);
      set({ error: 'Failed to load sample' });
    }
  },

  runVerification: async () => {
    const { inputText, pdfFile } = get();

    if (!inputText && !pdfFile) {
      set({ error: 'Please provide input text or upload a PDF' });
      return;
    }

    set({ isLoading: true, error: null });

    try {
      const formData = new FormData();

      if (pdfFile) {
        formData.append('file', pdfFile);
      } else if (inputText) {
        formData.append('text', inputText);
      }

      const response = await axios.post<VerificationResponse>(
        `${API_BASE_URL}/verify`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      );

      set({ results: response.data, isLoading: false });
    } catch (error) {
      console.error('Verification failed:', error);
      set({
        error: error instanceof Error ? error.message : 'Verification failed',
        isLoading: false
      });
    }
  },

  clearAll: () => set({
    inputText: '',
    pdfFile: null,
    selectedSample: null,
    results: null,
    error: null,
  }),
}));
