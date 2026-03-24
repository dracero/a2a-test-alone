/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '12000',
        pathname: '/message/file/**',
      },
    ],
  },
  async rewrites() {
    return [
      {
        source: '/conversation/:path*',
        destination: 'http://localhost:12000/conversation/:path*',
      },
      {
        source: '/message/:path*',
        destination: 'http://localhost:12000/message/:path*',
      },
      {
        // Configuración específica para archivos
        source: '/message/file/:id',
        destination: 'http://localhost:12000/message/file/:id',
      },
      {
        source: '/events/:path*',
        destination: 'http://localhost:12000/events/:path*',
      },
      {
        source: '/task/:path*',
        destination: 'http://localhost:12000/task/:path*',
      },
      {
        source: '/agent/:path*',
        destination: 'http://localhost:12000/agent/:path*',
      },
      {
        source: '/api_key/:path*',
        destination: 'http://localhost:12000/api_key/:path*',
      },
    ];
  },
  images: {
    remotePatterns: [
      {
        protocol: 'http',
        hostname: 'localhost',
        port: '12000',
        pathname: '/message/file/**',
      },
    ],
  },
};

module.exports = nextConfig;