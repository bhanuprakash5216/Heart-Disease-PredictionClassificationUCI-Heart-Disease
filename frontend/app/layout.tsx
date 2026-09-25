import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'HeartAI | Risk Analysis Lab',
  description: 'Educational machine learning demonstration for heart disease risk analysis.',
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
