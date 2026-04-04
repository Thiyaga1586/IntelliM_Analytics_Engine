from app.state_manager import StateManager

sm = StateManager()
with sm._conn() as conn:
    conn.execute("DELETE FROM predictions_log")
    conn.commit()

print("predictions_log cleared")