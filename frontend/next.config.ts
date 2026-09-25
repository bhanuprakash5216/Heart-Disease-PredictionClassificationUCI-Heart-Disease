import type { NextConfig } from 'next';
const nextConfig: NextConfig = {
	reactStrictMode: true,
	output: 'export',
	trailingSlash: true,
	basePath: process.env.NODE_ENV === 'production' ? '/Heart-Disease-PredictionClassificationUCI-Heart-Disease' : '',
	assetPrefix: process.env.NODE_ENV === 'production' ? '/Heart-Disease-PredictionClassificationUCI-Heart-Disease/' : '',
};
export default nextConfig;
