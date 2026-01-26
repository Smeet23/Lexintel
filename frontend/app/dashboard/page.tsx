'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useMutation } from '@tanstack/react-query';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';
import { FileUploadZone } from '@/components/file-upload-zone';
import apiClient from '@/lib/api';

export default function Dashboard() {
  const router = useRouter();
  const [caseName, setCaseName] = useState('');
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const uploadMutation = useMutation({
    mutationFn: async (formData: FormData) => {
      const response = await apiClient.post('/cases', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    },
    onSuccess: (data) => {
      router.push(`/cases/${data.id}`);
    },
    onError: (err: any) => {
      setError(err.response?.data?.detail || 'Failed to upload document');
    },
  });

  const handleFileSelect = (file: File) => {
    setSelectedFile(file);
    setError(null);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!caseName.trim()) {
      setError('Please enter a case name');
      return;
    }

    if (!selectedFile) {
      setError('Please select a file');
      return;
    }

    const formData = new FormData();
    formData.append('case_name', caseName);
    formData.append('file', selectedFile);

    uploadMutation.mutate(formData);
  };

  return (
    <div className="max-w-2xl mx-auto py-8">
      <h1 className="text-4xl font-bold mb-8">Upload Legal Document</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div>
          <label htmlFor="caseName" className="block text-sm font-medium text-gray-700 mb-2">
            Case Name
          </label>
          <Input
            id="caseName"
            type="text"
            value={caseName}
            onChange={(e) => setCaseName(e.target.value)}
            placeholder="Enter case name"
            disabled={uploadMutation.isPending}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Document
          </label>
          <FileUploadZone
            onFileSelect={handleFileSelect}
            isLoading={uploadMutation.isPending}
          />
        </div>

        {selectedFile && (
          <div className="bg-blue-50 border border-blue-200 rounded p-4">
            <p className="text-sm text-blue-800">
              <span className="font-medium">Selected file:</span> {selectedFile.name}
            </p>
            <p className="text-sm text-blue-800">
              <span className="font-medium">Size:</span> {(selectedFile.size / 1024 / 1024).toFixed(2)}MB
            </p>
          </div>
        )}

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Button
          type="submit"
          disabled={uploadMutation.isPending || !caseName.trim() || !selectedFile}
          className="w-full"
        >
          {uploadMutation.isPending ? 'Uploading...' : 'Upload & Analyze'}
        </Button>
      </form>
    </div>
  );
}
