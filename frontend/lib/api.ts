export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export type Patient = { age: number; sex: number; cp: number; trestbps: number; chol: number; fbs: number; restecg: number; thalach: number; exang: number; oldpeak: number; slope: number; ca: number; thal: number };
export type Prediction = { prediction: number; risk_probability: number; message: string; model: string; disclaimer: string };
export type Metrics = { selected_model: string; rows: number; metrics: Record<string, { accuracy: number; precision: number; recall: number; f1: number; roc_auc: number }> };

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options?.headers } });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? 'The API is unavailable. Start the FastAPI server and try again.');
  return response.json();
}
export const getMetrics = () => request<Metrics>('/model-metrics');
export const predict = (patient: Patient) => request<Prediction>('/predict', { method: 'POST', body: JSON.stringify(patient) });
