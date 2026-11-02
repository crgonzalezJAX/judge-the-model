# Decisions

<!-- Fill in each section at the step the activity website gives, then commit and push.
     Write each answer after its label. Keep the headings and labels exactly as they are:
     `python -m analysis` reads them. Limit: 150 words for Before results.
     Fill in Recommendations and Design choice and AI use last, once your app is built. -->

## Before results
For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.
- readmission: Positive class: readmitted within 30 days of discharge. A missed readmission matters more because a patient the model doesn't flag goes home without extra follow-up and may relapse with worse conditions. A false alarm only costs an unneeded follow-up call or visit, although flagging too many patients could increase costs and stretch hospital staff.
- diabetes: Positive class: has diabetes (doctor diagnosis or HbA1c ≥ 6.5%). A missed positive matters more because that person goes undiagnosed and untreated, with a higher chance of damage to the eyes, kidneys, nerves and heart. A false alarm only costs one extra follow-up blood test, a needle prick and some worry.

## Recommendations
- readmission: Logistic regression at threshold 0.10, with limits, because it catches 159 of 359 test readmissions (44%) while flagging 26 of every 100 patients, and it has the smallest training-to-test AUROC gap (0.657 training vs 0.640 test), so it did not memorize its training patients the way random forest did (0.997 training vs 0.651 test). Its test AUROC is still weak, only about 1 in 7 flagged patients is actually readmitted, and the threshold was chosen on test patients, so it should only trigger a low-cost follow-up such as a phone call and be rechecked on new patients first.
- diabetes: Logistic regression at threshold 0.12 because it meets the 90% sensitivity target on test patients (180 of 200 caught) with the fewest false alarms of any model (34.6 per 100 adults) and the highest test AUROC (0.827). The cost is that about half of adults get flagged, but a follow-up blood test is cheap, and test sensitivity is likely optimistic because the threshold was chosen while looking at test patients.

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
   - design choice: I made the threshold impossible to misread: a banner in Section 1 stating every number uses 0.50, and a plain-language slider label in Section 2 with a note that the cutoff is not the sensitivity, because while testing I mixed up the threshold with the sensitivity target myself, and a clinician is there for patients, not to troubleshoot the app.
- AI suggestion I turned down: For Section 1, Codex offered a "comparison first" layout that opens with the five-model table of six metrics. I chose "summary first" instead, like an executive summary that leads with the data points a clinician really wants (how many patients have the outcome, what "always no" scores, and how many the model actually catches) before the dense comparison table.
