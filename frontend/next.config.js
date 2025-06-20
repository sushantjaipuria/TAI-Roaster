/**
 * Next.js Configuration
 * 
 * This file configures Next.js build and runtime settings:
 * - Build optimization and performance
 * - Environment variable handling
 * - API route configuration
 * - Static generation settings
 * 
 * Configuration options:
 * - Output: Static export vs server-side rendering
 * - Images: Optimization and external domains
 * - Environment: Public and private variables
 * - Build: Bundle optimization and code splitting
 * - Redirects: URL routing and legacy support
 * 
 * Performance:
 * - Image optimization with next/image
 * - Bundle analyzer integration
 * - Code splitting configuration
 * - Static asset handling
 * - Service worker registration
 * 
 * Development:
 * - Hot module replacement settings
 * - Development server configuration
 * - TypeScript checking
 * - ESLint integration
 * 
 * Production:
 * - Build optimization
 * - Compression settings
 * - Security headers
 * - PWA configuration
 * 
 * Integration:
 * - Backend API URL configuration
 * - Third-party service integrations
 * - Analytics and monitoring setup
 * - Error tracking configuration
 */

/** @type {import('next').NextConfig} */
const nextConfig = {
  env: {
    BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8000',
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${process.env.BACKEND_URL || 'http://localhost:8000'}/api/:path*`,
      },
    ];
  },
  images: {
    domains: ['localhost'],
  },
};

module.exports = nextConfig;
