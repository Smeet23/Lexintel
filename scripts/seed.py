#!/usr/bin/env python3
"""
Seed script for LexIntel database.

Populates the database with realistic demo data for development and testing.
Run from project root:
    source backend/venv/bin/activate
    python scripts/seed.py

Or via justfile:
    just db-seed
"""

import sys
import os
import uuid
from datetime import datetime, timezone, timedelta

# Add project root to path so backend imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend"))

from dotenv import load_dotenv

# Load backend .env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), "backend", ".env"))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)


# ============================================
# Demo Data
# ============================================

NOW = datetime.now(timezone.utc)

MATTERS = [
    {
        "id": "10000000-0000-0000-0000-000000000001",
        "name": "Acme Corp vs. Global Industries - IP Dispute",
        "blob_storage_path": "matters/10000000-0000-0000-0000-000000000001/complaint.pdf",
        "file_type": "pdf",
        "status": "ready",
        "created_at": NOW - timedelta(days=14),
        "updated_at": NOW - timedelta(hours=2),
    },
    {
        "id": "10000000-0000-0000-0000-000000000002",
        "name": "Smith Estate Planning - Trust Agreement",
        "blob_storage_path": "matters/10000000-0000-0000-0000-000000000002/trust_agreement.pdf",
        "file_type": "pdf",
        "status": "ready",
        "created_at": NOW - timedelta(days=10),
        "updated_at": NOW - timedelta(hours=5),
    },
    {
        "id": "10000000-0000-0000-0000-000000000003",
        "name": "TechStart Inc. - Series A Investment Agreement",
        "blob_storage_path": "matters/10000000-0000-0000-0000-000000000003/investment_agreement.pdf",
        "file_type": "pdf",
        "status": "ready",
        "created_at": NOW - timedelta(days=7),
        "updated_at": NOW - timedelta(days=1),
    },
    {
        "id": "10000000-0000-0000-0000-000000000004",
        "name": "Metro Construction - Contractor Dispute",
        "blob_storage_path": "matters/10000000-0000-0000-0000-000000000004/contract.docx",
        "file_type": "docx",
        "status": "ready",
        "created_at": NOW - timedelta(days=3),
        "updated_at": NOW - timedelta(hours=12),
    },
    {
        "id": "10000000-0000-0000-0000-000000000005",
        "name": "Phoenix Merger - Due Diligence Review",
        "blob_storage_path": "matters/10000000-0000-0000-0000-000000000005/due_diligence.pdf",
        "file_type": "pdf",
        "status": "processing",
        "created_at": NOW - timedelta(hours=1),
        "updated_at": NOW - timedelta(minutes=30),
    },
    {
        "id": "10000000-0000-0000-0000-000000000006",
        "name": "Rivera Employment Agreement Review",
        "blob_storage_path": "matters/10000000-0000-0000-0000-000000000006/employment_agreement.pdf",
        "file_type": "pdf",
        "status": "ready",
        "created_at": NOW - timedelta(days=5),
        "updated_at": NOW - timedelta(days=2),
    },
    {
        "id": "10000000-0000-0000-0000-000000000007",
        "name": "DataFlow Inc. - GDPR Compliance Audit",
        "blob_storage_path": "matters/10000000-0000-0000-0000-000000000007/compliance_report.pdf",
        "file_type": "pdf",
        "status": "error",
        "created_at": NOW - timedelta(days=2),
        "updated_at": NOW - timedelta(days=1),
    },
]

# Sample chunks for the first few matters (simulate processed documents)
CHUNKS = []
chunk_texts = {
    "10000000-0000-0000-0000-000000000001": [
        ("1", "COMPLAINT FOR PATENT INFRINGEMENT\n\nPlaintiff Acme Corporation, by and through its attorneys, hereby brings this action against Global Industries Inc. for patent infringement under 35 U.S.C. § 271."),
        ("1", "PARTIES\n\n1. Plaintiff Acme Corporation is a Delaware corporation with its principal place of business at 100 Innovation Drive, San Jose, CA 95110.\n\n2. Defendant Global Industries Inc. is a corporation organized under the laws of the State of New York."),
        ("2", "JURISDICTION AND VENUE\n\n3. This Court has subject matter jurisdiction pursuant to 28 U.S.C. §§ 1331 and 1338(a).\n\n4. Venue is proper in this district under 28 U.S.C. §§ 1391 and 1400(b)."),
        ("3", "FACTUAL BACKGROUND\n\n5. Acme Corporation is the owner by assignment of U.S. Patent No. 11,234,567 ('the '567 Patent'), entitled 'System and Method for Automated Document Analysis Using Machine Learning.'"),
        ("4", "CLAIMS\n\n6. Global Industries' product known as 'DocuScan Pro' infringes at least Claims 1, 3, and 7 of the '567 Patent by implementing automated document classification using neural network architectures substantially similar to those described in the patent."),
        ("5", "DAMAGES\n\n7. Acme Corporation has suffered and continues to suffer damages as a result of Global Industries' infringement. Acme seeks compensatory damages in an amount no less than $15,000,000."),
    ],
    "10000000-0000-0000-0000-000000000002": [
        ("1", "TRUST AGREEMENT\n\nThis Trust Agreement ('Agreement') is entered into this 15th day of January, 2026, by and between James R. Smith ('Grantor') and First National Trust Company ('Trustee')."),
        ("2", "ARTICLE I - TRUST PROPERTY\n\nThe Grantor hereby transfers to the Trustee the property described in Schedule A attached hereto, to be held, administered, and distributed in accordance with the terms of this Agreement."),
        ("3", "ARTICLE II - BENEFICIARIES\n\nThe primary beneficiaries of this trust shall be the Grantor's children: Sarah Smith, Michael Smith, and Elizabeth Smith-Jones. Upon the death of the Grantor, the trust assets shall be distributed equally among the surviving beneficiaries."),
        ("4", "ARTICLE III - TRUSTEE POWERS\n\nThe Trustee shall have the power to invest and reinvest trust assets, collect income, pay expenses, and make distributions as outlined in this Agreement. The Trustee may employ investment advisors and legal counsel."),
        ("5", "ARTICLE IV - TAX PROVISIONS\n\nThis trust is intended to qualify as a grantor trust under IRC §§ 671-679. All income generated by the trust shall be reported on the Grantor's personal income tax return during the Grantor's lifetime."),
    ],
    "10000000-0000-0000-0000-000000000003": [
        ("1", "SERIES A PREFERRED STOCK PURCHASE AGREEMENT\n\nThis agreement is entered into as of February 1, 2026, by and among TechStart Inc., a Delaware corporation (the 'Company'), and the investors listed on Exhibit A (the 'Investors')."),
        ("2", "SECTION 1 - PURCHASE AND SALE\n\n1.1 The Company agrees to sell and the Investors agree to purchase an aggregate of 5,000,000 shares of Series A Preferred Stock at a price of $2.00 per share, for a total investment of $10,000,000."),
        ("3", "SECTION 2 - REPRESENTATIONS AND WARRANTIES\n\n2.1 The Company represents that it is duly organized and in good standing under the laws of the State of Delaware.\n2.2 The authorized capital stock consists of 50,000,000 shares of Common Stock and 10,000,000 shares of Preferred Stock."),
        ("4", "SECTION 3 - LIQUIDATION PREFERENCE\n\n3.1 In the event of any liquidation, dissolution, or winding up of the Company, holders of Series A Preferred Stock shall be entitled to receive, prior to any distribution to holders of Common Stock, an amount equal to 1.5x the original issue price."),
    ],
    "10000000-0000-0000-0000-000000000004": [
        ("1", "CONSTRUCTION CONTRACT\n\nThis Construction Contract ('Contract') is made effective as of January 1, 2026, between Metro Construction LLC ('Owner') and BuildRight Contractors Inc. ('Contractor')."),
        ("2", "SCOPE OF WORK\n\nThe Contractor shall furnish all labor, materials, equipment, and services necessary for the construction of a 12-story mixed-use commercial building located at 450 Main Street, Downtown District."),
        ("3", "CONTRACT PRICE AND PAYMENT\n\nThe total contract price is $28,500,000. Payment shall be made in monthly progress installments based on certified completion percentages. A 10% retainage shall be held until substantial completion."),
        ("4", "DISPUTE RESOLUTION\n\nAny disputes arising under this Contract shall first be submitted to mediation. If mediation fails, disputes shall be resolved by binding arbitration under the rules of the American Arbitration Association."),
    ],
}

for matter_id, texts in chunk_texts.items():
    for seq, (page, content) in enumerate(texts):
        CHUNKS.append({
            "id": str(uuid.uuid4()),
            "matter_id": matter_id,
            "page_num": page,
            "section_name": None,
            "content": content,
            "chunk_sequence": seq,
            "created_at": NOW - timedelta(days=7),
        })

# Sample queries for matters
QUERIES = [
    {
        "id": str(uuid.uuid4()),
        "matter_id": "10000000-0000-0000-0000-000000000001",
        "question": "What are the key patent claims being asserted?",
        "answer": "The complaint asserts infringement of Claims 1, 3, and 7 of U.S. Patent No. 11,234,567, titled 'System and Method for Automated Document Analysis Using Machine Learning.' The patent covers automated document classification using neural network architectures. The plaintiff alleges that Global Industries' 'DocuScan Pro' product implements substantially similar neural network architectures for document classification.",
        "citations": [
            {"page_num": "4", "content": "Claims 1, 3, and 7 of the '567 Patent", "relevance_score": 0.95},
            {"page_num": "3", "content": "U.S. Patent No. 11,234,567", "relevance_score": 0.88},
        ],
        "created_at": NOW - timedelta(days=12),
    },
    {
        "id": str(uuid.uuid4()),
        "matter_id": "10000000-0000-0000-0000-000000000001",
        "question": "What is the amount of damages being sought?",
        "answer": "Acme Corporation is seeking compensatory damages of no less than $15,000,000 for the patent infringement by Global Industries.",
        "citations": [
            {"page_num": "5", "content": "damages in an amount no less than $15,000,000", "relevance_score": 0.97},
        ],
        "created_at": NOW - timedelta(days=11),
    },
    {
        "id": str(uuid.uuid4()),
        "matter_id": "10000000-0000-0000-0000-000000000002",
        "question": "Who are the beneficiaries of the Smith trust?",
        "answer": "The primary beneficiaries are the Grantor's three children: Sarah Smith, Michael Smith, and Elizabeth Smith-Jones. Upon the death of the Grantor (James R. Smith), the trust assets are to be distributed equally among the surviving beneficiaries.",
        "citations": [
            {"page_num": "3", "content": "Sarah Smith, Michael Smith, and Elizabeth Smith-Jones", "relevance_score": 0.96},
        ],
        "created_at": NOW - timedelta(days=8),
    },
    {
        "id": str(uuid.uuid4()),
        "matter_id": "10000000-0000-0000-0000-000000000003",
        "question": "What is the liquidation preference for Series A investors?",
        "answer": "Holders of Series A Preferred Stock are entitled to a 1.5x liquidation preference. In the event of any liquidation, dissolution, or winding up of the Company, they will receive 1.5 times the original issue price ($2.00 per share = $3.00 per share) prior to any distribution to Common Stock holders.",
        "citations": [
            {"page_num": "4", "content": "1.5x the original issue price", "relevance_score": 0.94},
            {"page_num": "2", "content": "$2.00 per share, for a total investment of $10,000,000", "relevance_score": 0.82},
        ],
        "created_at": NOW - timedelta(days=5),
    },
    {
        "id": str(uuid.uuid4()),
        "matter_id": "10000000-0000-0000-0000-000000000004",
        "question": "How are disputes resolved under this contract?",
        "answer": "The contract provides for a two-step dispute resolution process: (1) disputes must first be submitted to mediation, and (2) if mediation fails to resolve the dispute, it proceeds to binding arbitration under the rules of the American Arbitration Association.",
        "citations": [
            {"page_num": "4", "content": "mediation... binding arbitration under the rules of the American Arbitration Association", "relevance_score": 0.93},
        ],
        "created_at": NOW - timedelta(days=2),
    },
]


def seed():
    """Insert all seed data."""
    session = Session()

    try:
        # Check if data already exists
        result = session.execute(text("SELECT COUNT(*) FROM matters"))
        count = result.scalar()
        if count and count > 0:
            print(f"Database already has {count} matters. Clearing existing data...")
            session.execute(text("DELETE FROM queries"))
            session.execute(text("DELETE FROM chunks"))
            session.execute(text("DELETE FROM processing_jobs"))
            session.execute(text("DELETE FROM matters"))
            session.commit()
            print("Cleared existing data.")

        # Insert matters
        for m in MATTERS:
            session.execute(text("""
                INSERT INTO matters (id, name, blob_storage_path, file_type, status, is_deleted, created_at, updated_at)
                VALUES (:id, :name, :blob_storage_path, :file_type, :status, false, :created_at, :updated_at)
            """), m)
        print(f"Inserted {len(MATTERS)} matters")

        # Insert chunks
        for c in CHUNKS:
            session.execute(text("""
                INSERT INTO chunks (id, matter_id, page_num, section_name, content, chunk_sequence, created_at)
                VALUES (:id, :matter_id, :page_num, :section_name, :content, :chunk_sequence, :created_at)
            """), c)
        print(f"Inserted {len(CHUNKS)} chunks")

        # Insert queries
        import json as _json
        for q in QUERIES:
            session.execute(text("""
                INSERT INTO queries (id, matter_id, question, answer, citations, created_at)
                VALUES (:id, :matter_id, :question, :answer, CAST(:citations AS jsonb), :created_at)
            """), {**q, "citations": _json.dumps(q["citations"])})
        print(f"Inserted {len(QUERIES)} queries")

        session.commit()
        print("\nSeed complete!")
        print(f"  - {len(MATTERS)} matters ({sum(1 for m in MATTERS if m['status'] == 'ready')} ready, {sum(1 for m in MATTERS if m['status'] == 'processing')} processing, {sum(1 for m in MATTERS if m['status'] == 'error')} error)")
        print(f"  - {len(CHUNKS)} document chunks")
        print(f"  - {len(QUERIES)} sample queries with citations")

    except Exception as e:
        session.rollback()
        print(f"Error seeding database: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    seed()
