# Implementation Plan: Frontend Integration

## Overview

This implementation plan integrates the Next.js frontend application with the existing Python backend system. The approach follows these phases:

1. Clone and configure the frontend repository
2. Set up npm orchestration scripts in the root package.json
3. Modify the backend for CORS support and file caching
4. Configure environment variables
5. Implement property-based tests for correctness validation
6. Perform end-to-end testing

The implementation ensures that a single `npm run dev` command starts all services: the FastAPI backend, multiple Python agents, and the Next.js frontend.

## Tasks

- [x]   1. Clone and configure frontend repository
    - [x] 1.1 Clone a2a-frontend repository into frontend/ directory
        - Clone the repository to `frontend/` at project root
        - Preserve all original files and directory structure
        - _Requirements: 1.1, 1.2_
    - [x] 1.2 Verify frontend dependencies and configuration
        - Confirm package.json has Next.js 13.5.1, React 18.2.0, TypeScript, Tailwind CSS
        - Verify all UI components exist (ChatInterface, AdminDashboard, etc.)
        - Check that API client is configured for localhost:12000
        - _Requirements: 1.3, 1.4, 1.5_

- [x]   2. Configure root package.json for process orchestration
    - [x] 2.1 Add concurrently dependency to root package.json
        - Install concurrently as devDependency
        - _Requirements: 7.3_
    - [x] 2.2 Create npm scripts for all services
        - Add dev:backend script to start FastAPI server
        - Add dev:frontend script to start Next.js app
        - Add dev:agent:images, dev:agent:medical, dev:agent:multimodal scripts
        - Add main dev script using concurrently with color-coded output
        - Add install:frontend, build:frontend, type-check scripts
        - _Requirements: 2.1, 2.2, 2.3, 2.4, 7.2_

- [x]   3. Modify backend for CORS and file caching
    - [x] 3.1 Add CORS middleware to FastAPI server
        - Import CORSMiddleware from fastapi.middleware.cors
        - Configure allow_origins with http://localhost:3000
        - Set allow_credentials, allow_methods, allow_headers
        - Add middleware before route handlers in demo/ui/main.py
        - _Requirements: 3.6_
    - [x] 3.2 Implement file caching in ConversationServer
        - Add \_file_cache dict to store file_id -> FilePart mappings
        - Add \_message_to_cache dict to store message_id:part_index -> file_id mappings
        - Implement cache_content() method to convert FileWithBytes to FileWithUri
        - Generate unique cache IDs using uuid4
        - _Requirements: 10.1, 10.2_
    - [x] 3.3 Implement file serving endpoint
        - Add GET /message/file/{file_id} endpoint
        - Retrieve file from \_file_cache using file_id
        - Decode base64 for image MIME types
        - Return file with correct Content-Type header
        - Return 404 if file_id not found
        - _Requirements: 10.3, 10.4, 10.5_
    - [x] 3.4 Modify message parsing to handle frontend format
        - Update parse_message_from_dict() to handle both FileWithBytes and FileWithUri
        - Implement restore_files_from_cache() to restore cached files when needed
        - Update \_send_message() to cache files before storing messages
        - Update \_list_messages() to replace FileWithBytes with FileWithUri in responses
        - _Requirements: 3.2, 3.3, 3.4, 3.5_

- [x]   4. Configure environment variables
    - [x] 4.1 Create or update .env file at project root
        - Set A2A_UI_HOST=0.0.0.0
        - Set A2A_UI_PORT=12000
        - Set A2A_HOST=BEEAI
        - Add GOOGLE_API_KEY placeholder
        - Add GOOGLE_GENAI_USE_VERTEXAI placeholder
        - _Requirements: 8.1, 8.2, 8.3, 8.5, 8.6_
    - [x] 4.2 Verify frontend API client configuration
        - Confirm baseURL is set to http://localhost:12000 in frontend/src/lib/api/client.ts
        - _Requirements: 8.4_

- [x]   5. Checkpoint - Verify basic integration
    - Ensure all tests pass, ask the user if questions arise.

- [ ]   6. Implement property-based tests for backend
    - [ ]\* 6.1 Write property test for message round trip
        - **Property 1: Message Send-Receive Round Trip**
        - **Validates: Requirements 3.2, 4.3, 4.7**
        - Use Hypothesis to generate random messages with text and file parts
        - Send message via /message/send, retrieve via /message/list
        - Verify text parts unchanged, file parts converted to FileWithUri
        - Test file: demo/ui/tests/test_message_round_trip.py
    - [ ]\* 6.2 Write property test for file caching uniqueness
        - **Property 2: File Caching Uniqueness**
        - **Validates: Requirements 10.1**
        - Use Hypothesis to generate multiple unique binary files
        - Send each file in a message, extract cache IDs from responses
        - Verify all cache IDs are unique
        - Test file: demo/ui/tests/test_file_caching.py
    - [ ]\* 6.3 Write property test for file caching round trip
        - **Property 3: File Caching Round Trip**
        - **Validates: Requirements 3.5, 10.2, 10.4, 10.5**
        - Use Hypothesis to generate random binary data and MIME types
        - Upload file, retrieve via /message/file/{file_id}
        - Verify content and MIME type match original
        - Test file: demo/ui/tests/test_file_caching.py
    - [ ]\* 6.4 Write property test for agent auto-registration
        - **Property 6: Agent Auto-Registration**
        - **Validates: Requirements 6.5**
        - Use Hypothesis to generate random agent port numbers
        - Start mock agents on those ports
        - Verify all agents appear in /agent/list response
        - Test file: demo/ui/tests/test_agent_registration.py

- [ ]   7. Implement property-based tests for frontend
    - [ ]\* 7.1 Write property test for image base64 encoding
        - **Property 4: Image Base64 Encoding**
        - **Validates: Requirements 4.5**
        - Use fast-check to generate random binary image data
        - Encode to base64, verify valid base64 format
        - Decode and verify matches original data
        - Test file: frontend/src/lib/utils/**tests**/file-utils.test.ts
    - [ ]\* 7.2 Write property test for file URI resolution
        - **Property 8: File URI Resolution**
        - **Validates: Requirements 10.3**
        - Use fast-check to generate random UUIDs as file IDs
        - Verify resolveFileUrl() constructs correct URL format
        - Test file: frontend/src/lib/utils/**tests**/file-utils.test.ts
    - [ ]\* 7.3 Write property test for API error display
        - **Property 9: API Error Display**
        - **Validates: Requirements 11.4**
        - Use fast-check to generate random HTTP error codes and messages
        - Simulate API errors in ChatInterface component
        - Verify error alert is displayed with user-friendly message
        - Test file: frontend/src/components/chat/**tests**/ChatInterface.test.tsx

- [ ]   8. Implement unit tests for backend
    - [ ]\* 8.1 Write unit tests for backend server startup
        - Test server starts on port 12000
        - Test CORS middleware allows requests from http://localhost:3000
        - Test environment variables default correctly
        - Test file: demo/ui/tests/test_server_config.py
    - [ ]\* 8.2 Write unit tests for API endpoints
        - Test /conversation/list returns conversations
        - Test /message/list returns messages for conversation
        - Test /message/pending returns pending notifications
        - Test /agent/list returns registered agents
        - Test /message/file/{file_id} returns 404 for non-existent ID
        - Test /message/file/{file_id} returns correct MIME type
        - Test file: demo/ui/tests/test_api_endpoints.py

- [ ]   9. Implement unit tests for frontend
    - [ ]\* 9.1 Write unit tests for API client
        - Test API client uses correct base URL (http://localhost:12000)
        - Test error handling for network failures
        - Test error handling for API errors
        - Test file: frontend/src/lib/api/**tests**/client.test.ts
    - [ ]\* 9.2 Write unit tests for chat interface
        - Test chat interface renders text input and image upload button
        - Test file upload validates file size (max 10MB)
        - Test file upload validates file type (images only)
        - Test error notification displays on upload failure
        - Test file: frontend/src/components/chat/**tests**/ChatInterface.test.tsx
    - [ ]\* 9.3 Write unit tests for admin dashboard
        - Test admin dashboard renders all 7 navigation tabs
        - Test clicking each tab navigates to correct route
        - Test file: frontend/src/app/admin/**tests**/layout.test.tsx
    - [ ]\* 9.4 Write unit tests for polling manager
        - Test polling starts when component mounts
        - Test polling stops when component unmounts
        - Test polling interval is 2 seconds for normal operation
        - Test polling interval increases to 5 seconds after error
        - Test file: frontend/src/lib/utils/**tests**/polling.test.ts
    - [ ]\* 9.5 Write unit test for TypeScript configuration
        - Test TypeScript build completes without type errors
        - Test tsconfig.json has strict mode enabled
        - Test file: frontend/**tests**/typescript-config.test.ts

- [ ]   10. Checkpoint - Verify all tests pass
    - Ensure all tests pass, ask the user if questions arise.

- [ ]   11. Perform end-to-end integration testing
    - [ ]\* 11.1 Test complete system startup
        - Start all processes with npm run dev
        - Verify backend accessible at http://localhost:12000
        - Verify frontend accessible at http://localhost:3000
        - Verify all 3 agents running on ports 10001-10003
        - Test file: e2e/test_system_startup.spec.ts
    - [ ]\* 11.2 Test message sending and receiving
        - Send text message from frontend
        - Verify message appears in conversation view
        - Upload image from frontend
        - Verify image displays correctly in chat
        - Test file: e2e/test_messaging.spec.ts
    - [ ]\* 11.3 Test admin dashboard functionality
        - Navigate to each admin dashboard tab
        - Verify data loads correctly in each tab
        - Test file: e2e/test_admin_dashboard.spec.ts
    - [ ]\* 11.4 Test error handling and recovery
        - Stop one agent process
        - Verify error handling displays correctly
        - Restart agent
        - Verify agent re-registers successfully
        - Test file: e2e/test_error_handling.spec.ts

- [ ]   12. Final checkpoint - Complete integration verification
    - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at key milestones
- Property tests validate universal correctness properties across all inputs
- Unit tests validate specific examples and edge cases
- End-to-end tests validate complete user workflows
- The frontend uses TypeScript with strict mode for type safety
- The backend uses Python with FastAPI for the REST API
- All processes are orchestrated with npm + concurrently for simplified development
