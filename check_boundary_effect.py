from src.faithfulness import _nli_scores

claim = ("After a student is notified that programme-fee payments remain outstanding, "
         "the University will give them 5 working days to settle the outstanding balance.")

broken = ("er 1 June 2027 for those whose programmes start in January 2027), where a student "
          "fails to make the necessary programme fee payment(s) in accordance with the remainder "
          "of their relevant published payment plan, the University will notify the student in "
          "writing and give them 5 working days' notice to settle the outstanding account balance.")

clean = ("Where a student fails to make the necessary programme fee payment(s) in accordance with "
         "their published payment plan, the University will notify the student in writing and give "
         "them 5 working days' notice to settle the outstanding account balance.")

for name, premise in [("as-chunked", broken), ("repaired", clean)]:
    ent, neu, con = _nli_scores(premise, claim)
    print(f"{name:12s} ent={ent:.3f} neu={neu:.3f} con={con:.3f}")
