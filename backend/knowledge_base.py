"""
knowledge_base.py
-----------------
Predefined response knowledge base for auto-resolution.
Each response has a usefulness weight h(n) used in A*-inspired scoring.
These are the "goal nodes" that the A* search selects from.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Knowledge Base: category → list of {response, keywords, h_score}
# h_score = heuristic usefulness weight (higher = more universally useful)
# ─────────────────────────────────────────────────────────────────────────────

KNOWLEDGE_BASE = {
    "Billing": [
        {
            "id": "B001",
            "title": "Duplicate Charge Resolution",
            "response": (
                "We sincerely apologize for the duplicate charge on your account. "
                "Our billing team has been notified and will issue a full refund within 3–5 business days. "
                "You will receive a confirmation email once processed. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["charged twice", "double charge", "duplicate", "payment twice", "deducted twice"],
            "h_score": 0.90
        },
        {
            "id": "B002",
            "title": "Unauthorized Charge Investigation",
            "response": (
                "We take unauthorized charges very seriously. "
                "We have flagged your account for immediate review by our billing security team. "
                "Any unauthorized amount will be reversed within 48 hours. "
                "You will also receive a detailed statement breakdown via email. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["unauthorized", "strange charge", "mystery charge", "unknown fee", "misc fee"],
            "h_score": 0.88
        },
        {
            "id": "B003",
            "title": "Refund Processing",
            "response": (
                "Your refund request has been approved and is being processed immediately. "
                "Refunds typically appear in your account within 5–7 business days depending on your bank. "
                "If you don't see it within 7 days, please contact us with reference: #AUTO-{ticket_id}"
            ),
            "keywords": ["refund", "money back", "return payment", "reimbursement", "credit"],
            "h_score": 0.85
        },
        {
            "id": "B004",
            "title": "Incorrect Bill Correction",
            "response": (
                "We've reviewed your billing concern and identified the discrepancy. "
                "A corrected invoice will be sent to your registered email within 24 hours. "
                "Any overpayment will be applied as a credit to your next billing cycle. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["wrong bill", "incorrect invoice", "wrong amount", "overcharged", "billing error"],
            "h_score": 0.87
        },
        {
            "id": "B005",
            "title": "Subscription Cancellation Billing",
            "response": (
                "We've noted that you were charged after cancellation. "
                "This should not happen and we apologize for the inconvenience. "
                "The post-cancellation charge will be fully refunded within 3 business days. "
                "Your account status has been updated to reflect the correct cancellation date. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["charged after cancel", "still being charged", "cancelled but charged", "post-cancellation"],
            "h_score": 0.86
        },
        {
            "id": "B006",
            "title": "General Billing Assistance",
            "response": (
                "Thank you for reaching out about your billing concern. "
                "Our billing team has been alerted and will review your account within 24 hours. "
                "You can also view your detailed billing history in the 'Billing' section of your account portal. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["bill", "charge", "payment", "invoice", "fee", "price"],
            "h_score": 0.75
        }
    ],

    "Technical": [
        {
            "id": "T001",
            "title": "Internet Connectivity Fix",
            "response": (
                "We've detected potential connectivity issues in your area. "
                "Please try these steps: (1) Restart your modem/router by unplugging for 30 seconds. "
                "(2) Check all cable connections. (3) Run our diagnostic tool at diagnostics.resolveai.com. "
                "If the issue persists after these steps, a technician visit will be scheduled. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["no internet", "connection dropping", "internet down", "no connectivity", "disconnecting"],
            "h_score": 0.92
        },
        {
            "id": "T002",
            "title": "Slow Speed Resolution",
            "response": (
                "We're sorry your speeds aren't meeting expectations. "
                "We've run a remote diagnostic on your line and identified potential congestion. "
                "A line optimization has been applied remotely — please restart your router. "
                "If speeds remain below 80% of your plan speed, we will dispatch a technician within 48 hours. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["slow speed", "slow internet", "buffering", "lagging", "slower than expected", "upload speed"],
            "h_score": 0.89
        },
        {
            "id": "T003",
            "title": "App / Portal Error Fix",
            "response": (
                "We've identified a known issue with our app/portal affecting some users. "
                "Our engineering team has deployed a fix. Please: (1) Clear your browser cache and cookies. "
                "(2) Try an incognito/private window. (3) Update the app to the latest version. "
                "This should resolve error codes and login issues immediately. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["app crash", "error code", "portal error", "login error", "503", "dashboard error"],
            "h_score": 0.88
        },
        {
            "id": "T004",
            "title": "Device / Equipment Support",
            "response": (
                "We've logged your device issue. "
                "Please perform a factory reset by holding the reset button for 10 seconds. "
                "Then run the setup wizard again. If the firmware update failed, "
                "visit support.resolveai.com/firmware to manually install the latest version. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["device", "modem", "router", "equipment", "firmware", "smart tv", "not connecting"],
            "h_score": 0.85
        },
        {
            "id": "T005",
            "title": "Network Outage Notification",
            "response": (
                "We have confirmed a network outage in your area that our team is actively working to resolve. "
                "Estimated restoration time: within 4 hours. "
                "You will receive an SMS/email notification when service is restored. "
                "We apologize for the disruption and will apply a service credit to your account. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["outage", "network down", "no service", "area outage", "complete outage"],
            "h_score": 0.91
        },
        {
            "id": "T006",
            "title": "General Technical Support",
            "response": (
                "Thank you for reporting this technical issue. "
                "Our technical support team has been notified and will begin remote diagnostics on your connection. "
                "You may receive a call from our tech team within 2 hours. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["technical", "not working", "problem", "issue", "broken", "error"],
            "h_score": 0.72
        }
    ],

    "Service": [
        {
            "id": "S001",
            "title": "Escalation to Senior Support",
            "response": (
                "We sincerely apologize for the poor service experience you've had. "
                "Your complaint has been escalated to our Senior Customer Relations team. "
                "A dedicated case manager will contact you within 4 hours to personally handle your issue. "
                "We are committed to making this right. Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["escalate", "rude", "unprofessional", "ignored", "no response", "manager"],
            "h_score": 0.93
        },
        {
            "id": "S002",
            "title": "Callback Scheduling",
            "response": (
                "We apologize that you haven't received your promised callback. "
                "A priority callback has been scheduled for you within the next 2 hours. "
                "Our agent will have full context of your case before calling. "
                "We are also crediting your account for the inconvenience caused. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["callback", "no call", "no response", "promised call", "waited", "never called"],
            "h_score": 0.88
        },
        {
            "id": "S003",
            "title": "Ticket Follow-Up",
            "response": (
                "We apologize that your previous ticket was not followed up on promptly. "
                "Your ticket has been flagged as high-priority and assigned to a specialist. "
                "You will receive a status update within 6 hours. "
                "We value your patience and are committed to resolving this quickly. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["ticket", "no update", "still waiting", "unresolved", "not resolved", "follow up"],
            "h_score": 0.86
        },
        {
            "id": "S004",
            "title": "Wait Time Apology",
            "response": (
                "We are deeply sorry for the excessive wait time you experienced. "
                "This is not the standard of service we aim to provide. "
                "Your feedback has been logged and sent to our Quality Assurance team. "
                "To make up for this, we've applied a service credit to your account. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["hold", "wait", "45 minutes", "hours", "long wait", "on hold", "waiting"],
            "h_score": 0.84
        },
        {
            "id": "S005",
            "title": "General Service Complaint Resolution",
            "response": (
                "Thank you for bringing this service concern to our attention. "
                "We take all service feedback very seriously. "
                "Your complaint has been documented and forwarded to the relevant department manager. "
                "A representative will reach out to you within 24 hours with a resolution. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["service", "support", "agent", "representative", "staff", "customer service"],
            "h_score": 0.76
        }
    ],

    "General": [
        {
            "id": "G001",
            "title": "Account Management Info",
            "response": (
                "Great question! You can manage your account easily through our portal at app.resolveai.com. "
                "Features available include: billing management, plan upgrades/downgrades, "
                "security settings (including 2FA), and usage statistics. "
                "Our mobile app is available on iOS and Android. Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["account", "manage", "update", "portal", "app", "mobile", "settings"],
            "h_score": 0.82
        },
        {
            "id": "G002",
            "title": "Plans and Pricing Info",
            "response": (
                "We offer several plans to suit different needs: Basic (50Mbps), Standard (200Mbps), "
                "and Premium (1Gbps). All plans include 24/7 support and free equipment. "
                "Student and senior discounts of 20% are available with valid ID. "
                "Visit pricing.resolveai.com for current promotions. Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["plans", "pricing", "discount", "deal", "promotion", "basic", "premium", "student", "senior"],
            "h_score": 0.83
        },
        {
            "id": "G003",
            "title": "Cancellation and Refund Policy",
            "response": (
                "You can cancel your subscription at any time with no penalty during the first 30 days. "
                "After 30 days, a 30-day notice period applies. "
                "To cancel, visit app.resolveai.com/cancel or call our cancellation line. "
                "Your data will be securely deleted within 30 days of account closure. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["cancel", "cancellation", "close account", "terminate", "refund policy", "exit"],
            "h_score": 0.84
        },
        {
            "id": "G004",
            "title": "General FAQ Response",
            "response": (
                "Thank you for your inquiry! For immediate answers, our comprehensive FAQ is available at "
                "help.resolveai.com. Our support team is available Monday–Friday 8AM–8PM and "
                "Saturday 9AM–5PM. You can also use our 24/7 AI chat for instant help. "
                "Reference ID: #AUTO-{ticket_id}"
            ),
            "keywords": ["question", "information", "how", "what", "when", "policy", "hours"],
            "h_score": 0.75
        }
    ]
}


def get_responses_for_category(category: str) -> list:
    """Return all responses for a given category."""
    return KNOWLEDGE_BASE.get(category, [])


def get_all_responses() -> list:
    """Return all responses across all categories as a flat list."""
    all_responses = []
    for category, responses in KNOWLEDGE_BASE.items():
        for r in responses:
            all_responses.append({**r, "category": category})
    return all_responses
