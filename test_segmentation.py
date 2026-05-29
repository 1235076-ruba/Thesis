import process_requirements as pr

nlp = pr.load_nlp()
tests = [
    ("search", "The search feature must return results matching the user text query within 1.5 seconds and highlight the matching keywords in bold."),
    ("onboarding", "A first-time user must be able to complete the onboarding tutorial and create their first project within 5 minutes without external assistance."),
    ("backup", "The database shall automatically backup every hour and send alert logs."),
    ("timeout", 'When a network timeout occurs, the system must display an error message stating: "Connection lost. Please check your internet and try again," without crashing the application.'),
    ("beautiful", "The user interface should be beautiful and modern."),
    (
        "lead_in",
        "When operating in offline mode or during low battery states, the system must save state data and notify users.",
    ),
]
for name, t in tests:
    segs = pr.segment_requirement(t, nlp)
    print(name, pr.format_segmentation(segs) or "ATOMIC")
    for s in segs:
        print(" ", pr.is_valid_requirement_segment(s, nlp), s)
    print()
