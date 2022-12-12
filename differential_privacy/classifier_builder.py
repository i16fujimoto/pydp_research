from sklearn.ensemble import RandomForestClassifier

def build_classifier():
  rfc = RandomForestClassifier(random_state=42)
  return rfc