#!/usr/bin/env python3
"""
Simple test script to verify the complete RAG workflow:
1. Upload a PDF
2. Wait for processing
3. Query the processed document
"""

import asyncio
import time
import requests
import json
from pathlib import Path

# Configuration
API_URL = "http://localhost:8000"
TEST_PDF_PATH = Path("/tmp/test_document.pdf")

def create_test_pdf():
    """Create a simple test PDF"""
    # Minimal valid PDF
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>
endobj
4 0 obj
<< /Length 44 >>
stream
BT
/F1 12 Tf
100 700 Td
(This is a test document.) Tj
ET
endstream
endobj
5 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
xref
0 6
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000229 00000 n
0000000323 00000 n
trailer
<< /Size 6 /Root 1 0 R >>
startxref
406
%%EOF"""

    TEST_PDF_PATH.write_bytes(pdf_content)
    print(f"✓ Created test PDF: {TEST_PDF_PATH}")
    return TEST_PDF_PATH

def test_health():
    """Test API health"""
    print("\n[1] Testing API health...")
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200, f"Health check failed: {response.text}"
    print(f"✓ API is healthy: {response.json()}")

def test_upload_pdf():
    """Upload a test PDF"""
    print("\n[2] Uploading test PDF...")

    with open(TEST_PDF_PATH, 'rb') as f:
        files = {'file': f}
        data = {'name': 'Test Case Document'}
        response = requests.post(
            f"{API_URL}/cases",
            files=files,
            data=data
        )

    assert response.status_code == 200, f"Upload failed: {response.text}"
    result = response.json()
    case_id = result['id']
    task_id = result.get('task_id', 'unknown')
    print(f"✓ PDF uploaded successfully!")
    print(f"  Case ID: {case_id}")
    print(f"  Task ID: {task_id}")
    print(f"  Status: {result['status']}")
    return case_id

def test_check_status(case_id):
    """Check processing status"""
    print(f"\n[3] Checking case status...")
    response = requests.get(f"{API_URL}/cases/{case_id}/status")
    assert response.status_code == 200, f"Status check failed: {response.text}"
    result = response.json()
    print(f"✓ Case status: {result['status']}")
    return result

def wait_for_processing(case_id, max_wait=60):
    """Wait for case to finish processing"""
    print(f"\n[4] Waiting for document processing (max {max_wait}s)...")
    start_time = time.time()

    while time.time() - start_time < max_wait:
        status_result = test_check_status(case_id)

        if status_result['status'] == 'ready':
            print(f"✓ Document processing complete!")
            return True
        elif status_result['status'] == 'error':
            print(f"✗ Document processing failed!")
            return False

        print(f"  Status: {status_result['status']}, waiting...")
        time.sleep(2)

    print(f"✗ Processing timeout!")
    return False

def test_query(case_id):
    """Query the processed document"""
    print(f"\n[5] Querying the document...")

    question = "What is this document about?"
    response = requests.post(
        f"{API_URL}/cases/{case_id}/ask",
        json={"question": question}
    )

    assert response.status_code == 200, f"Query failed: {response.text}"
    result = response.json()

    print(f"✓ Query successful!")
    print(f"  Question: {question}")
    print(f"  Answer: {result.get('answer', 'N/A')[:200]}...")
    print(f"  Confidence: {result.get('confidence', 'N/A')}")
    print(f"  Sources: {len(result.get('sources', []))} documents")

    if result.get('error'):
        print(f"  Error: {result['error']}")

    return result

def main():
    """Run the complete workflow test"""
    print("=" * 60)
    print("LexIntel RAG System - Workflow Test")
    print("=" * 60)

    try:
        # Setup
        create_test_pdf()

        # Test API
        test_health()

        # Upload
        case_id = test_upload_pdf()

        # Wait for processing
        if wait_for_processing(case_id):
            # Query
            test_query(case_id)
            print("\n" + "=" * 60)
            print("✓ Complete workflow test PASSED!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ Processing failed or timed out")
            print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
