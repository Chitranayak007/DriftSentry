import networkx as nx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="DriftSentry Backend API")

# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# MOCK DATABASE
# =========================================================

USERS = {
    "alice": {
        "id": "alice",
        "name": "Alice",
        "email": "alice@example.com",
        "team": "Engineering",
        "role": "Developer",
        "ip": "10.0.0.12",
        "location": "US",
    },

    "bob": {
        "id": "bob",
        "name": "Bob",
        "email": "bob@example.com",
        "team": "Engineering",
        "role": "Developer",
        "ip": "192.168.1.20",
        "location": "US",
    },

    "carol": {
        "id": "carol",
        "name": "Carol",
        "email": "carol@example.com",
        "team": "Engineering",
        "role": "Developer",
        "ip": "185.220.101.45",
        "location": "DE",
    },
}


# =========================================================
# SYNTHETIC EVENTS
# =========================================================

EVENTS = {
    "alice": [
        {
            "timestamp": "2026-09-01T10:00:00",
            "machine": "dev-01",
            "resource": "repo-atlas",
            "action": "READ",
            "project": "Atlas",
        },
        {
            "timestamp": "2026-09-07T11:00:00",
            "machine": "dev-02",
            "resource": "repo-titan",
            "action": "READ",
            "project": "Titan",
        },
        {
            "timestamp": "2026-09-10T12:00:00",
            "machine": "dev-03",
            "resource": "titan-db",
            "action": "READ",
            "project": "Titan",
        },
    ],

    "bob": [
        {
            "timestamp": "2026-09-01T10:00:00",
            "machine": "dev-01",
            "resource": "repo-atlas",
            "action": "READ",
            "project": "Atlas",
        },
        {
            "timestamp": "2026-09-07T22:30:00",
            "machine": "unknown-42",
            "resource": "repo-finance",
            "action": "READ",
            "project": "Finance",
        },
        {
            "timestamp": "2026-09-14T23:30:00",
            "machine": "unknown-42",
            "resource": "finance-db",
            "action": "READ",
            "project": "Finance",
        },
    ],

    "carol": [
        {
            "timestamp": "2026-09-01T10:00:00",
            "machine": "dev-03",
            "resource": "repo-atlas",
            "action": "READ",
            "project": "Atlas",
        },
        {
            "timestamp": "2026-09-10T03:00:00",
            "machine": "foreign-machine",
            "resource": "repo-sensitive",
            "action": "DOWNLOAD",
            "project": "Sensitive",
        },
    ],
}


# =========================================================
# CONTEXT DATA
# =========================================================

CONTEXT = {
    "alice": {
        "hr": True,
        "jira": True,
        "github": True,
        "justification_found": True,
        "reason": "Alice was assigned to Project Titan.",
    },

    "bob": {
        "hr": False,
        "jira": False,
        "github": False,
        "justification_found": False,
        "reason": None,
    },

    "carol": {
        "hr": False,
        "jira": False,
        "github": False,
        "justification_found": False,
        "reason": None,
    },
}


# =========================================================
# PLACEHOLDER ML RESULT
# =========================================================
# ML GUY CAN REPLACE THIS FUNCTION LATER
# =========================================================

from ml_engine import run_ml_analysis


# =========================================================
# PLACEHOLDER GRAPH RESULT
# =========================================================
# GRAPH GIRL CAN REPLACE THIS FUNCTION LATER
# =========================================================

def run_graph_analysis(user_id: str, events: list) -> dict:

    # If there are no events, return an empty graph result
    if not events:
        return {
            "graph_drift_score": 0.0,
            "new_nodes": [],
            "new_relationships": []
        }

    # -----------------------------------------------------
    # 1. Create the graph
    # -----------------------------------------------------

    graph = nx.MultiDiGraph()

    user = USERS.get(user_id, {})

    # Add user node
    graph.add_node(
        f"user:{user_id}",
        type="user",
        label=user_id
    )

    # Add IP node
    if user.get("ip"):
        graph.add_node(
            f"ip:{user['ip']}",
            type="ip",
            label=user["ip"]
        )

        graph.add_edge(
            f"user:{user_id}",
            f"ip:{user['ip']}",
            relation="USED_IP"
        )

    # Add location node
    if user.get("location"):
        graph.add_node(
            f"location:{user['location']}",
            type="location",
            label=user["location"]
        )

        graph.add_edge(
            f"user:{user_id}",
            f"location:{user['location']}",
            relation="LOCATED_IN"
        )

    # -----------------------------------------------------
    # 2. Sort events by time
    # -----------------------------------------------------

    events = sorted(
        events,
        key=lambda event: event.get("timestamp", "")
    )

    # -----------------------------------------------------
    # 3. First event = user's baseline
    # -----------------------------------------------------

    baseline_event = events[0]

    baseline_nodes = set()

    for key in ["machine", "resource", "project"]:

        value = baseline_event.get(key)

        if value:
            baseline_nodes.add(
                (key, str(value))
            )

    # -----------------------------------------------------
    # 4. Store new nodes and relationships
    # -----------------------------------------------------

    new_nodes = []
    new_relationships = []

    # -----------------------------------------------------
    # 5. Analyse later events
    # -----------------------------------------------------

    event_scores = []

    for event in events[1:]:

        machine = str(event.get("machine", "")).strip()
        resource = str(event.get("resource", "")).strip()
        project = str(event.get("project", "")).strip()
        action = str(event.get("action", "")).upper()

        # ---------------------------------------------
        # Add nodes to graph
        # ---------------------------------------------

        if machine:

            graph.add_node(
                f"machine:{machine}",
                type="machine",
                label=machine
            )

            graph.add_edge(
                f"user:{user_id}",
                f"machine:{machine}",
                relation="USED"
            )

        if resource:

            graph.add_node(
                f"resource:{resource}",
                type="resource",
                label=resource
            )

            graph.add_edge(
                f"user:{user_id}",
                f"resource:{resource}",
                relation="ACCESSED"
            )

        if project:

            graph.add_node(
                f"project:{project}",
                type="project",
                label=project
            )

            graph.add_edge(
                f"user:{user_id}",
                f"project:{project}",
                relation="MEMBER_OF"
            )

        # Machine -> Resource relationship
        if machine and resource:

            graph.add_edge(
                f"machine:{machine}",
                f"resource:{resource}",
                relation="HOSTS"
            )

        # Resource -> Project relationship
        if resource and project:

            graph.add_edge(
                f"resource:{resource}",
                f"project:{project}",
                relation="BELONGS_TO"
            )

        # ---------------------------------------------
        # Detect new nodes
        # ---------------------------------------------

        current_nodes = [
            ("machine", machine),
            ("resource", resource),
            ("project", project)
        ]

        for node_type, value in current_nodes:

            if not value:
                continue

            if (node_type, value) not in baseline_nodes:

                if value not in new_nodes:
                    new_nodes.append(value)

        # ---------------------------------------------
        # Detect new relationships
        # ---------------------------------------------

        relationships = [
            machine,
            resource,
            project
        ]

        for value in relationships:

            if not value:
                continue

            baseline_values = [
                str(baseline_event.get("machine", "")),
                str(baseline_event.get("resource", "")),
                str(baseline_event.get("project", ""))
            ]

            if value not in baseline_values:

                relationship = f"{user_id} -> {value}"

                if relationship not in new_relationships:
                    new_relationships.append(
                        relationship
                    )

        # -------------------------------------------------
        # 6. Calculate suspiciousness of this event
        # -------------------------------------------------

        event_score = 0.0

        # New machine
        if machine and ("machine", machine) not in baseline_nodes:
            event_score += 0.05

        # New resource
        if resource and ("resource", resource) not in baseline_nodes:
            event_score += 0.10

        # New project
        if project and ("project", project) not in baseline_nodes:
            event_score += 0.05

        # Sensitive resources/projects
        sensitive_words = [
            "finance",
            "sensitive",
            "secret",
            "confidential",
            "payroll",
            "production",
            "prod",
            "admin"
        ]

        text_to_check = (
            resource.lower() + " " + project.lower()
        )

        if any(
            word in text_to_check
            for word in sensitive_words
        ):
            event_score += 0.35

        # Unknown/external machines
        suspicious_machine_words = [
            "unknown",
            "foreign",
            "external"
        ]

        if any(
            word in machine.lower()
            for word in suspicious_machine_words
        ):
            event_score += 0.20

        # Suspicious actions
        if action in ["DOWNLOAD", "EXPORT", "DELETE"]:

            event_score += 0.15

        # User location outside expected region
        if user.get("location") not in ["US", "IN"]:

            event_score += 0.15

        event_scores.append(
            min(1.0, event_score)
        )

    # -----------------------------------------------------
    # 7. Calculate final graph drift score
    # -----------------------------------------------------

    if event_scores:

        graph_drift_score = sum(event_scores) / len(event_scores)

    else:

        graph_drift_score = 0.0

    # Repeated suspicious behaviour is more important
    # than one isolated event.
    suspicious_events = sum(
        1 for score in event_scores
        if score >= 0.50
    )

    if suspicious_events >= 2:

        graph_drift_score += 0.05

    graph_drift_score = round(
        min(1.0, graph_drift_score),
        2
    )

    # -----------------------------------------------------
    # 8. Return the format expected by the backend
    # -----------------------------------------------------

    return {
        "graph_drift_score": graph_drift_score,
        "new_nodes": new_nodes,
        "new_relationships": new_relationships
    }

# =========================================================
# CONTEXT GUARD
# =========================================================

def evaluate_context(user_id: str) -> dict:

    context = CONTEXT.get(user_id)

    if context is None:
        return {
            "justification_found": False,
            "reason": None
        }

    return context


# =========================================================
# RISK ENGINE
# =========================================================

def calculate_risk(
    ml_result: dict,
    graph_result: dict,
    context: dict,
    user_id: str
) -> dict:

    ml_score = ml_result["anomaly_score"] * 50
    graph_score = graph_result["graph_drift_score"] * 50

    risk_score = ml_score + graph_score

    # Context correction
    if context["justification_found"]:
        risk_score -= 40

    # Foreign IP signal
    if USERS[user_id]["location"] != "US":
        risk_score += 15

    risk_score = round(max(0, min(100, risk_score)))

    if risk_score >= 80:
        risk_level = "Critical"

    elif risk_score >= 50:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return {
        "score": risk_score,
        "level": risk_level
    }


# =========================================================
# TIMELINE
# =========================================================

def generate_timeline(user_id: str, final_score: int) -> list:

    if user_id == "alice":
        scores = [18, 24, 29, final_score]

    elif user_id == "bob":
        scores = [20, 38, 62, final_score]

    elif user_id == "carol":
        scores = [18, 31, 67, final_score]

    else:
        scores = [10, 15, 20, final_score]

    return [
        {"day": 1, "score": scores[0]},
        {"day": 7, "score": scores[1]},
        {"day": 14, "score": scores[2]},
        {"day": 21, "score": scores[3]},
    ]


# =========================================================
# EVIDENCE
# =========================================================

def generate_evidence(
    ml_result: dict,
    graph_result: dict
) -> list:

    evidence = []

    for signal in ml_result["signals"]:
        evidence.append({
            "source": "ML",
            "reason": signal
        })

    for relationship in graph_result["new_relationships"]:
        evidence.append({
            "source": "Graph",
            "reason": relationship
        })

    return evidence


# =========================================================
# SOC STORYLINE GENERATOR
# =========================================================

def _risk_action(level: str, justification_found: bool) -> str:
    if justification_found:
        return "Continue monitoring; the available context provides a legitimate explanation for the observed drift."
    if level == "Critical":
        return "Initiate analyst investigation and review the user's recent access and account activity."
    if level == "Medium":
        return "Review the activity and continue monitoring for repeated or escalating drift."
    return "Continue monitoring for additional behavioral changes."


def generate_soc_storyline(
    user: dict,
    risk: dict,
    ml_result: dict,
    graph_result: dict,
    context: dict
) -> dict:
    """Generate an explainable SOC narrative without an external LLM/API.

    The storyline is built from the actual ML signals, graph changes, and
    ContextGuard result, so the dashboard remains fully functional offline.
    """
    signals = list(ml_result.get("signals", []))
    relationships = list(graph_result.get("new_relationships", []))

    # Keep the narrative concise while preserving the strongest evidence.
    signal_text = "; ".join(signals[:3])
    graph_text = "; ".join(relationships[:2])

    if not signal_text:
        signal_text = "No major rule-based behavioral signals were identified."

    if graph_text:
        graph_sentence = f" Graph analysis identified {graph_text}."
    else:
        graph_sentence = " No new suspicious graph relationships were identified."

    if context.get("justification_found"):
        context_sentence = (
            f" ContextGuard found legitimate context: {context.get('reason') or 'activity is contextually justified'}."
        )
    else:
        context_sentence = " ContextGuard found no legitimate justification in the available HR/Jira/GitHub context."

    summary = (
        f"{user['name']} shows {risk['level'].lower()} behavioral risk with a "
        f"risk score of {risk['score']}/100. Key indicators include {signal_text}."
        f"{graph_sentence}{context_sentence}"
    )

    return {
        "summary": summary,
        "assessment": f"{risk['level']} behavioral risk.",
        "recommendation": _risk_action(
            risk["level"], context.get("justification_found", False)
        ),
        "generator": "Deterministic SOC Storyline Generator",
        "api_key_required": False,
    }


# =========================================================
# ROUTES
# =========================================================

@app.get("/")
def root():

    return {
        "name": "DriftSentry Backend",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.get("/users")
def get_users():

    return list(USERS.values())


@app.get("/users/{name}")
def get_user(name: str):

    user = USERS.get(name.lower())

    if user is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{name}' not found"
        )

    return user


@app.get("/events/{name}")
def get_events(name: str):

    name = name.lower()

    if name not in USERS:
        raise HTTPException(
            status_code=404,
            detail=f"User '{name}' not found"
        )

    return {
        "user": name,
        "events": EVENTS.get(name, [])
    }


# =========================================================
# MAIN ANALYSIS ENDPOINT
# =========================================================

@app.get("/analyze/{name}")
def analyze(name: str):

    name = name.lower()

    user = USERS.get(name)

    if user is None:
        raise HTTPException(
            status_code=404,
            detail=f"User '{name}' not found"
        )

    events = EVENTS.get(name, [])

    # 1. ML
    ml_result = run_ml_analysis(
        name,
        events
    )

    # 2. Graph
    graph_result = run_graph_analysis(
        name,
        events
    )

    # 3. Context
    context = evaluate_context(name)

    # 4. Risk
    risk = calculate_risk(
        ml_result,
        graph_result,
        context,
        name
    )

    # 5. Timeline
    timeline = generate_timeline(
        name,
        risk["score"]
    )

    # 6. Evidence
    evidence = generate_evidence(
        ml_result,
        graph_result
    )

    # 7. SOC storyline
    ai_result = generate_soc_storyline(
        user,
        risk,
        ml_result,
        graph_result,
        context
    )

    return {
        "user": user,

        "risk": risk,

        "ml": ml_result,

        "graph": graph_result,

        "context": context,

        "timeline": timeline,

        "evidence": evidence,

        "ai": ai_result
    }


# =========================================================
# TIMELINE ENDPOINT
# =========================================================

@app.get("/timeline/{name}")
def get_timeline(name: str):

    result = analyze(name)

    return {
        "user": name.lower(),
        "timeline": result["timeline"]
    }


# =========================================================
# GRAPH ENDPOINT
# =========================================================

@app.get("/graph/{name}")
def get_graph(name: str):

    name = name.lower()

    if name not in USERS:
        raise HTTPException(
            status_code=404,
            detail=f"User '{name}' not found"
        )

    return run_graph_analysis(
        name,
        EVENTS.get(name, [])
    )


# =========================================================
# INVESTIGATION ENDPOINT
# =========================================================

@app.get("/investigate/{name}")
def investigate(name: str):

    result = analyze(name)

    return {
        "user": result["user"],
        "risk": result["risk"],
        "evidence": result["evidence"],
        "context": result["context"],
        "investigation": result["ai"]
    }