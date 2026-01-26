'use client';

import { useState, useEffect } from 'react';
import { useParams } from 'next/navigation';
import { useQuery, useMutation } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle, Copy, CheckCircle } from 'lucide-react';
import { SpinnerWithText } from '@/components/spinner';
import apiClient from '@/lib/api';

interface CaseStatus {
  id: string;
  status: 'processing' | 'ready' | 'error';
  error?: string;
}

interface QueryAnswer {
  answer: string;
  sources: Array<{
    page: number;
    text: string;
  }>;
}

export default function CaseDetail() {
  const params = useParams();
  const caseId = params.id as string;
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<QueryAnswer | null>(null);
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null);

  // Polling query for case status
  const { data: caseStatus, isLoading: isCheckingStatus } = useQuery<CaseStatus>({
    queryKey: ['caseStatus', caseId],
    queryFn: async () => {
      const response = await apiClient.get(`/cases/${caseId}/status`);
      return response.data;
    },
    refetchInterval: (query) => {
      // Stop polling when status is not 'processing'
      if (query.state.data?.status !== 'processing') {
        return false;
      }
      return 2000; // Poll every 2 seconds
    },
    refetchIntervalInBackground: false,
  });

  const askMutation = useMutation({
    mutationFn: async (q: string) => {
      const response = await apiClient.post(`/cases/${caseId}/ask`, { question: q });
      return response.data;
    },
    onSuccess: (data) => {
      setAnswer(data);
      setQuestion('');
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    askMutation.mutate(question);
  };

  const handleCopySource = (text: string, index: number) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  if (isCheckingStatus) {
    return (
      <div className="max-w-4xl mx-auto py-8 flex justify-center">
        <SpinnerWithText text="Loading case details..." />
      </div>
    );
  }

  if (!caseStatus) {
    return (
      <div className="max-w-4xl mx-auto py-8">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>Failed to load case</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (caseStatus.status === 'processing') {
    return (
      <div className="max-w-4xl mx-auto py-8 flex justify-center">
        <SpinnerWithText text="Processing document... This may take a few minutes." />
      </div>
    );
  }

  if (caseStatus.status === 'error') {
    return (
      <div className="max-w-4xl mx-auto py-8">
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{caseStatus.error || 'An error occurred while processing the document'}</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto py-8">
      <h1 className="text-4xl font-bold mb-8">Case Details</h1>

      <div className="bg-white rounded-lg shadow p-6 space-y-6">
        <div>
          <h2 className="text-2xl font-semibold mb-4">Ask LexIntel</h2>
          <form onSubmit={handleSubmit} className="space-y-4">
            <Textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask a question about the legal document..."
              disabled={askMutation.isPending}
              rows={4}
            />

            <Button
              type="submit"
              disabled={askMutation.isPending || !question.trim()}
              className="w-full"
            >
              {askMutation.isPending ? 'Processing...' : 'Ask LexIntel'}
            </Button>

            {askMutation.isError && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>
                  {(askMutation.error as any)?.response?.data?.detail || 'Failed to process question'}
                </AlertDescription>
              </Alert>
            )}
          </form>
        </div>

        {answer && (
          <div className="mt-8 space-y-6">
            <div>
              <h3 className="text-xl font-semibold mb-3">Answer</h3>
              <div className="bg-blue-50 border border-blue-200 rounded p-4">
                <p className="text-gray-800 whitespace-pre-wrap">{answer.answer}</p>
              </div>
            </div>

            {answer.sources && answer.sources.length > 0 && (
              <div>
                <h3 className="text-xl font-semibold mb-3">Sources & Citations</h3>
                <div className="space-y-3">
                  {answer.sources.map((source, index) => (
                    <div key={index} className="bg-gray-50 border border-gray-200 rounded p-4">
                      <div className="flex justify-between items-start mb-2">
                        <p className="text-sm font-medium text-gray-600">
                          Page {source.page}
                        </p>
                        <button
                          onClick={() => handleCopySource(source.text, index)}
                          className="flex items-center gap-1 text-blue-600 hover:text-blue-700 text-sm"
                        >
                          {copiedIndex === index ? (
                            <>
                              <CheckCircle className="h-4 w-4" />
                              <span>Copied</span>
                            </>
                          ) : (
                            <>
                              <Copy className="h-4 w-4" />
                              <span>Copy</span>
                            </>
                          )}
                        </button>
                      </div>
                      <p className="text-gray-700 text-sm">{source.text}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
