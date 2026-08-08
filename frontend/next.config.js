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
        destination: 'http://127.0.0.1:12000/conversation/:path*',
      },
      {
        source: '/message/:path*',
        destination: 'http://127.0.0.1:12000/message/:path*',
      },
      {
        // Configuración específica para archivos
        source: '/message/file/:id',
        destination: 'http://127.0.0.1:12000/message/file/:id',
      },
      {
        source: '/events/:path*',
        destination: 'http://127.0.0.1:12000/events/:path*',
      },
      {
        source: '/task/:path*',
        destination: 'http://127.0.0.1:12000/task/:path*',
      },
      {
        source: '/agent/:path*',
        destination: 'http://127.0.0.1:12000/agent/:path*',
      },
      {
        source: '/api_key/:path*',
        destination: 'http://127.0.0.1:12000/api_key/:path*',
      },
      {
        source: '/nams/:path*',
        destination: 'http://127.0.0.1:12000/nams/:path*',
      },
      {
        source: '/correct',
        destination: 'http://127.0.0.1:12000/correct',
      },
    ];
  },
};

module.exports = nextConfig;