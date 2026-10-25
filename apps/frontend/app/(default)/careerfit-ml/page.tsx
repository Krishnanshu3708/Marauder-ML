'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import {
  ArrowLeft,
  Brain,
  CheckCircle,
  WarningCircle,
  ArrowsClockwise,
  Sparkle,
  Target,
  FileText,
  Briefcase,
  Lightning,
} from '@phosphor-icons/react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Alert } from '@/components/ui/alert';
import { PageFrame } from '@/components/ui/page-frame';
import { PageHeader } from '@/components/ui/page-header';
import { Card, CardHeader, CardTitle } from '@/components/ui/card';
import {
  analyzeResumeJobMatch,
  type MatchAnalysisResponse,
} from '@/lib/api/careerfit-ml';

// Preset samples for immediate demonstration and testing
const SAMPLES = {
  pythonBackend: {
    resume: `Senior Backend Software Engineer with 5 years experience designing scalable REST APIs and distributed microservices using Python, FastAPI, PostgreSQL, Redis, and Docker. Implemented automated testing with Pytest, CI/CD pipelines via GitHub Actions, and containerized deployments on AWS. Optimized SQL query execution plans reducing database query latency by 40%.`,
    job: `Seeking Senior Python Developer to architect and maintain backend services. Required skills: Python, FastAPI or Django, PostgreSQL, Docker, and experience with cloud infrastructure (AWS). Hands-on experience with RESTful API design, relational databases, and microservices is required.`,
  },
  dataScience: {
    resume: `Data Scientist and Machine Learning Engineer with 4 years experience in Python, Scikit-Learn, PyTorch, Pandas, and SQL. Built end-to-end NLP text classification pipelines, regression models, and exploratory data analysis dashboards. Experienced deploying models via FastAPI and Docker on GCP.`,
    job: `We are hiring an Applied Machine Learning Engineer. Requirements: Strong proficiency in Python, Scikit-Learn, PyTorch or TensorFlow, Pandas, and SQL. Experience building NLP or predictive models and deploying them into production environments.`,
  },
  crossDomainMismatch: {
    resume: `Certified Public Accountant (CPA) with 7 years experience in corporate tax auditing, balance sheet reconciliation, QuickBooks, and financial risk assessment. Deep knowledge of US GAAP compliance, financial reporting, and payroll management.`,
    job: `Looking for Senior DevOps and Cloud Infrastructure Engineer. Required skills: Kubernetes, Docker, Terraform, CI/CD automation, Linux system administration, and AWS cloud architecture.`,
  },
};

export default function MarauderMLPage() {
  const [resumeText, setResumeText] = useState('');
  const [jobDescription, setJobDescription] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisResult, setAnalysisResult] = useState<MatchAnalysisResponse | null>(null);

  const handleRunAnalysis = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!resumeText.trim()) {
      setError('Please provide candidate resume text before running analysis.');
      return;
    }
    if (!jobDescription.trim()) {
      setError('Please provide target job description text before running analysis.');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await analyzeResumeJobMatch({
        resume_text: resumeText.trim(),
        job_description: jobDescription.trim(),
      });
      setAnalysisResult(response);
    } catch (err) {
      console.error('Marauder ML analysis failed:', err);
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to complete analysis. Please ensure the backend server is running.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const loadPreset = (presetKey: keyof typeof SAMPLES) => {
    setResumeText(SAMPLES[presetKey].resume);
    setJobDescription(SAMPLES[presetKey].job);
    setError(null);
    setAnalysisResult(null);
  };

  const clearInputs = () => {
    setResumeText('');
    setJobDescription('');
    setError(null);
    setAnalysisResult(null);
  };

  return (
    <div className="cf-careerfit-theme">
      <PageFrame width="wide" className="cf-careerfit-frame">
      <PageHeader>
        <Link
          href="/dashboard"
          className="mb-8 inline-flex items-center gap-2 border border-ink bg-canvas px-3 py-1 font-mono text-xs font-bold uppercase shadow-sw-sm transition hover:bg-canvas-subtle"
        >
          <ArrowLeft aria-hidden="true" />
          Back to Dashboard
        </Link>
        <PageHeader.Title>Marauder ML</PageHeader.Title>
        <PageHeader.Subtitle>
          I Solemnly Swear I Am Qualified — Resume Matching, TF-IDF Text Similarity & Skill Gap Analysis
        </PageHeader.Subtitle>
      </PageHeader>

      <div className="p-6 md:p-10 space-y-8 bg-canvas">
        {/* Academic Overview Notice */}
        <div className="border border-ink bg-white p-6 shadow-sw-sm">
          <div className="flex items-start gap-3">
            <Brain aria-hidden="true" className="size-6 text-primary shrink-0 mt-0.5" />
            <div className="space-y-1">
              <h2 className="font-mono text-sm font-bold uppercase text-ink">
                Marauder ML Mini-Project Pipeline
              </h2>
              <p className="font-sans text-sm text-ink-soft leading-relaxed">
                This module executes quantitative machine learning algorithms locally: TF-IDF
                vectorization, Cosine Similarity, technical skill taxonomy gap analysis, and a
                supervised Logistic Regression classification model. All processing is local without
                external LLM dependencies.
              </p>
            </div>
          </div>
        </div>

        {error && (
          <Alert tone="error" title="Input / Analysis Error">
            {error}
          </Alert>
        )}

        {/* Input Form Section */}
        <form onSubmit={handleRunAnalysis} className="space-y-6">
          {/* Quick Preset Buttons */}
          <div className="flex flex-wrap items-center gap-2 pb-2">
            <span className="font-mono text-xs font-bold uppercase text-steel mr-2">
              Sample Inputs:
            </span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => loadPreset('pythonBackend')}
            >
              <Sparkle aria-hidden="true" />
              Python Backend Match
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => loadPreset('dataScience')}
            >
              <Sparkle aria-hidden="true" />
              Data Science Match
            </Button>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => loadPreset('crossDomainMismatch')}
            >
              <WarningCircle aria-hidden="true" />
              Accounting Mismatch
            </Button>
            {(resumeText || jobDescription) && (
              <Button type="button" variant="ghost" size="sm" onClick={clearInputs}>
                Clear All
              </Button>
            )}
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Resume Text Input */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="resume-text" className="flex items-center gap-2 font-mono text-xs uppercase font-bold text-ink">
                  <FileText aria-hidden="true" className="size-4" />
                  Candidate Resume Text
                </Label>
                <span className="font-mono text-xs text-steel">
                  {resumeText.length} characters
                </span>
              </div>
              <Textarea
                id="resume-text"
                value={resumeText}
                onChange={(e) => setResumeText(e.target.value)}
                placeholder="Paste candidate resume, experience bullets, and skills here..."
                rows={12}
                className="font-mono text-xs leading-relaxed"
                disabled={isLoading}
              />
            </div>

            {/* Job Description Input */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="job-description" className="flex items-center gap-2 font-mono text-xs uppercase font-bold text-ink">
                  <Briefcase aria-hidden="true" className="size-4" />
                  Target Job Description
                </Label>
                <span className="font-mono text-xs text-steel">
                  {jobDescription.length} characters
                </span>
              </div>
              <Textarea
                id="job-description"
                value={jobDescription}
                onChange={(e) => setJobDescription(e.target.value)}
                placeholder="Paste job posting, required qualifications, and responsibilities here..."
                rows={12}
                className="font-mono text-xs leading-relaxed"
                disabled={isLoading}
              />
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <Button
              type="submit"
              variant="default"
              size="lg"
              disabled={isLoading || !resumeText.trim() || !jobDescription.trim()}
              className="min-w-56"
            >
              {isLoading ? (
                <>
                  <ArrowsClockwise aria-hidden="true" className="animate-spin" />
                  Analyzing Features...
                </>
              ) : (
                <>
                  <Lightning aria-hidden="true" />
                  Run Marauder ML Analysis
                </>
              )}
            </Button>
          </div>
        </form>

        {/* Results Section */}
        {analysisResult && (
          <div className="space-y-6 pt-6 border-t border-ink">
            <div className="flex items-center justify-between">
              <h2 className="font-serif text-3xl font-bold uppercase tracking-tight text-ink">
                Analysis Results & Insights
              </h2>
              <span className="font-mono text-xs font-bold uppercase bg-ink text-white px-3 py-1">
                Local ML Pipeline
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {/* Metric 1: TF-IDF Cosine Similarity */}
              <Card className="border border-ink bg-white shadow-sw-nested">
                <CardHeader>
                  <CardTitle className="text-sm font-mono uppercase text-steel flex items-center justify-between">
                    <span>Text Similarity</span>
                    <span className="text-xs px-2 py-0.5 bg-canvas border border-ink text-ink font-bold">
                      {analysisResult.match_category} Match
                    </span>
                  </CardTitle>
                  <div className="pt-2">
                    <p className="font-serif text-5xl font-bold text-ink">
                      {analysisResult.similarity_score.toFixed(1)}%
                    </p>
                    <p className="font-mono text-xs text-steel mt-1">
                      Raw Cosine: {analysisResult.raw_cosine.toFixed(4)}
                    </p>
                  </div>
                </CardHeader>
                <div className="px-6 pb-6 pt-2 text-xs font-sans text-ink-soft border-t border-slate-100">
                  Computed via TF-IDF n-gram vectorization and cosine angle projection between documents.
                </div>
              </Card>

              {/* Metric 2: Skill Gap Coverage */}
              <Card className="border border-ink bg-white shadow-sw-nested">
                <CardHeader>
                  <CardTitle className="text-sm font-mono uppercase text-steel flex items-center justify-between">
                    <span>Skill Coverage</span>
                    <span className="text-xs px-2 py-0.5 bg-canvas border border-ink text-ink font-bold">
                      {analysisResult.readiness_level}
                    </span>
                  </CardTitle>
                  <div className="pt-2">
                    <p className="font-serif text-5xl font-bold text-ink">
                      {analysisResult.skill_match_percentage.toFixed(1)}%
                    </p>
                    <p className="font-mono text-xs text-steel mt-1">
                      {analysisResult.matched_skills.length} matched /{' '}
                      {analysisResult.matched_skills.length + analysisResult.missing_skills.length} required
                    </p>
                  </div>
                </CardHeader>
                <div className="px-6 pb-6 pt-2 text-xs font-sans text-ink-soft border-t border-slate-100">
                  Extracted and normalized against 6 canonical tech skill taxonomies and aliases.
                </div>
              </Card>

              {/* Metric 3: Supervised Model Prediction */}
              <Card className="border border-ink bg-white shadow-sw-nested">
                <CardHeader>
                  <CardTitle className="text-sm font-mono uppercase text-steel flex items-center justify-between">
                    <span>Supervised ML Prediction</span>
                    <span
                      className={`text-xs px-2 py-0.5 border font-bold uppercase ${
                        analysisResult.prediction?.is_match
                          ? 'bg-emerald-50 border-emerald-500 text-emerald-800'
                          : analysisResult.prediction
                          ? 'bg-rose-50 border-rose-500 text-rose-800'
                          : 'bg-amber-50 border-amber-500 text-amber-800'
                      }`}
                    >
                      {analysisResult.prediction
                        ? analysisResult.prediction.is_match
                          ? 'Fit (Class 1)'
                          : 'Mismatch (Class 0)'
                        : 'Model Inactive'}
                    </span>
                  </CardTitle>
                  <div className="pt-2">
                    {analysisResult.prediction ? (
                      <>
                        <p className="font-serif text-5xl font-bold text-ink">
                          {(analysisResult.prediction.match_probability * 100).toFixed(1)}%
                        </p>
                        <p className="font-mono text-xs text-steel mt-1">
                          Estimated Fit Probability ({analysisResult.prediction.model_type})
                        </p>
                      </>
                    ) : (
                      <>
                        <p className="font-serif text-3xl font-bold text-steel pt-1">
                          N/A
                        </p>
                        <p className="font-mono text-xs text-steel mt-1">
                          Artifact Not Found
                        </p>
                      </>
                    )}
                  </div>
                </CardHeader>
                <div className="px-6 pb-6 pt-2 text-xs font-sans text-ink-soft border-t border-slate-100">
                  {analysisResult.prediction
                    ? 'Inferred by fitted Logistic Regression classifier over paired text TF-IDF features.'
                    : 'Train model via CLI to activate binary classification inference.'}
                </div>
              </Card>
            </div>

            {/* Metric Distinction Disclaimer */}
            <div className="bg-canvas border border-ink p-4 text-xs font-sans text-ink-soft">
              <strong className="font-mono uppercase font-bold text-ink">Important Evaluation Note: </strong>
              The TF-IDF Similarity Score represents geometric vector overlap between resume and job
              vocabulary. It is an unsupervised similarity metric, distinct from the supervised
              Logistic Regression model prediction, and must not be confused with training or test accuracy.
            </div>

            {/* Model Notes or Inactive Warning */}
            {analysisResult.notes && (
              <Alert tone={analysisResult.model_status === 'model_loaded' ? 'info' : 'warning'} title="Model Pipeline Status">
                {analysisResult.notes}
              </Alert>
            )}

            {/* Detailed Skill Gap & Vocabulary Breakdown */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Technical Skill Breakdown */}
              <div className="border border-ink bg-white p-6 shadow-sw-sm space-y-4">
                <h3 className="font-mono text-sm font-bold uppercase text-ink flex items-center gap-2">
                  <Target aria-hidden="true" className="size-4 text-primary" />
                  Skill Gap Breakdown
                </h3>

                {/* Matched Skills */}
                <div className="space-y-2">
                  <p className="font-mono text-xs font-bold uppercase text-emerald-800 flex items-center gap-1.5">
                    <CheckCircle aria-hidden="true" className="size-4 text-emerald-600" />
                    Matched Skills ({analysisResult.matched_skills.length})
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {analysisResult.matched_skills.length > 0 ? (
                      analysisResult.matched_skills.map((skill) => (
                        <span
                          key={skill}
                          className="inline-block bg-emerald-50 border border-emerald-300 text-emerald-900 font-mono text-xs px-2.5 py-1 uppercase"
                        >
                          {skill}
                        </span>
                      ))
                    ) : (
                      <span className="font-mono text-xs text-steel italic">
                        No overlapping skills detected.
                      </span>
                    )}
                  </div>
                </div>

                {/* Missing Skills */}
                <div className="space-y-2 pt-2 border-t border-slate-100">
                  <p className="font-mono text-xs font-bold uppercase text-rose-800 flex items-center gap-1.5">
                    <WarningCircle aria-hidden="true" className="size-4 text-rose-600" />
                    Missing Required Skills ({analysisResult.missing_skills.length})
                  </p>
                  <div className="flex flex-wrap gap-1.5">
                    {analysisResult.missing_skills.length > 0 ? (
                      analysisResult.missing_skills.map((skill) => (
                        <span
                          key={skill}
                          className="inline-block bg-rose-50 border border-rose-300 text-rose-900 font-mono text-xs px-2.5 py-1 uppercase"
                        >
                          {skill}
                        </span>
                      ))
                    ) : (
                      <span className="font-mono text-xs text-emerald-700 italic">
                        All identified job requirements are covered!
                      </span>
                    )}
                  </div>
                </div>

                {/* Additional Skills */}
                {analysisResult.additional_skills.length > 0 && (
                  <div className="space-y-2 pt-2 border-t border-slate-100">
                    <p className="font-mono text-xs font-bold uppercase text-steel">
                      Additional Candidate Skills ({analysisResult.additional_skills.length})
                    </p>
                    <div className="flex flex-wrap gap-1.5">
                      {analysisResult.additional_skills.map((skill) => (
                        <span
                          key={skill}
                          className="inline-block bg-canvas border border-slate-300 text-slate-700 font-mono text-xs px-2 py-0.5 uppercase"
                        >
                          {skill}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Informative Shared Vocabulary Terms */}
              <div className="border border-ink bg-white p-6 shadow-sw-sm space-y-4">
                <h3 className="font-mono text-sm font-bold uppercase text-ink flex items-center gap-2">
                  <Sparkle aria-hidden="true" className="size-4 text-primary" />
                  Top Shared TF-IDF Terms
                </h3>
                <p className="text-xs font-sans text-ink-soft">
                  These high-weight n-grams represent the most influential terms shared between the
                  candidate resume and job posting:
                </p>
                <div className="flex flex-wrap gap-2 pt-2">
                  {analysisResult.top_shared_terms.length > 0 ? (
                    analysisResult.top_shared_terms.map((term, index) => (
                      <span
                        key={term}
                        className="inline-flex items-center gap-1.5 bg-canvas border border-ink text-ink font-mono text-xs px-3 py-1.5"
                      >
                        <span className="text-steel font-bold">#{index + 1}</span>
                        <span>{term}</span>
                      </span>
                    ))
                  ) : (
                    <span className="font-mono text-xs text-steel italic">
                      No significant shared terms found.
                    </span>
                  )}
                </div>

                {/* Probability Distribution Box */}
                {analysisResult.prediction && (
                  <div className="pt-4 border-t border-slate-100 space-y-2">
                    <p className="font-mono text-xs font-bold uppercase text-ink">
                      Classifier Class Probabilities:
                    </p>
                    <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                      <div className="border border-slate-200 p-2.5 bg-slate-50">
                        <span className="text-steel block">P(Class 1 - Fit):</span>
                        <span className="font-bold text-ink text-base">
                          {(analysisResult.prediction.class_probabilities['1'] * 100 || 0).toFixed(1)}%
                        </span>
                      </div>
                      <div className="border border-slate-200 p-2.5 bg-slate-50">
                        <span className="text-steel block">P(Class 0 - Mismatch):</span>
                        <span className="font-bold text-ink text-base">
                          {(analysisResult.prediction.class_probabilities['0'] * 100 || 0).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
      </PageFrame>
    </div>
  );
}
