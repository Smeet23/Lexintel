export interface User {
  id: string;
  email: string;
  name?: string;
  created_at?: string;
  updated_at?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface Case {
  id: string;
  title: string;
  description?: string;
  case_number?: string;
  user_id: string;
  created_at: string;
  updated_at: string;
}

export interface Citation {
  id: string;
  case_id: string;
  text: string;
  source?: string;
  page_number?: number;
  created_at: string;
}

export interface QueryResponse {
  id: string;
  case_id: string;
  query: string;
  results: string[];
  created_at: string;
}

export interface UploadResponse {
  id: string;
  case_id: string;
  filename: string;
  size: number;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  created_at: string;
}
