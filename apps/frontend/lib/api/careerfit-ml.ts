/**
 * API client functions for Marauder ML endpoints.
 */

import { apiFetch } from '@/lib/api/client';

export interface ModelPredictionDetail {
  predicted_label: number;
  is_match: boolean;
  match_probability: number;
  class_probabilities: Record<string, number>;
  model_type: string;
}

export interface MatchAnalysisResponse {
  similarity_score: number;
  raw_cosine: number;
  match_category: string;
  top_shared_terms: string[];
  matched_skills: string[];
  missing_skills: string[];
  additional_skills: string[];
  skill_match_percentage: number;
  readiness_level: string;
  prediction: ModelPredictionDetail | null;
  model_status: 'model_loaded' | 'model_unavailable' | 'model_error';
  notes: string | null;
}

export interface MatchAnalysisRequest {
  resume_text: string;
  job_description: string;
  model_path?: string;
}

export interface MLModelStatusResponse {
  model_available: boolean;
  model_path: string;
  model_type: string;
  metadata: {
    timestamp_utc?: string;
    total_training_samples?: number;
    positive_samples?: number;
    negative_samples?: number;
    vocabulary_size?: number;
    model_type?: string;
  } | null;
}

/**
 * Call Marauder ML endpoint to perform TF-IDF similarity, skill gap, and classifier inference.
 */
export async function analyzeResumeJobMatch(
  request: MatchAnalysisRequest
): Promise<MatchAnalysisResponse> {
  const response = await apiFetch('/api/v1/ml/match-analysis', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    let errorMessage = `Analysis request failed with status ${response.status}`;
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorMessage = typeof errorJson.detail === 'string'
          ? errorJson.detail
          : JSON.stringify(errorJson.detail);
      }
    } catch {
      // Use fallback error message
    }
    throw new Error(errorMessage);
  }

  return response.json();
}

/**
 * Fetch status of the trained Marauder ML model.
 */
export async function fetchMLModelStatus(): Promise<MLModelStatusResponse> {
  const response = await apiFetch('/api/v1/ml/status', {
    method: 'GET',
  });

  if (!response.ok) {
    throw new Error(`Failed to fetch ML model status: ${response.status}`);
  }

  return response.json();
}
