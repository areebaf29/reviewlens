"""Generate data/demo_reviews_synthetic.csv: 400 SYNTHETIC reviews for a fictional product (NovaBuds X earbuds).
Synthetic on purpose: no real customer data. The `aspect` column is the ground truth used by the eval."""
import csv, random

random.seed(7)
ASPECTS = {
 "battery": (["Battery lasts a full work day and then some.", "I get around eight hours per charge, great for travel.", "Charging case gives three extra top ups."],
             ["Battery drains way too fast, barely four hours.", "The earbuds die after a couple of hours of music.", "Battery life dropped badly after two months."]),
 "noise": (["Noise cancelling blocks out the train and office chatter.", "ANC is excellent, I can't hear the fan at all.", "Great isolation on my commute."],
           ["Noise cancelling is weak, I still hear traffic.", "ANC makes a hissing sound and barely cuts background noise.", "Transparency mode sounds muffled and unnatural."]),
 "comfort": (["Very comfortable, I forget I'm wearing them.", "Lightweight fit that stays in during runs.", "The ear tips seal well and never hurt."],
             ["They hurt my ears after an hour.", "The fit is loose and they fall out when I walk fast.", "Too bulky, uncomfortable for sleeping or long calls."]),
 "connectivity": (["Bluetooth pairs instantly and stays connected.", "Switching between phone and laptop is seamless.", "Connection is rock solid across rooms."],
                  ["Keeps disconnecting from my phone every few minutes.", "Pairing fails half the time and I have to reset them.", "Audio stutters and cuts out in crowded places."]),
 "price": (["Great value for the money.", "Cheaper than the big brands and just as good.", "Worth every rupee."],
           ["Overpriced for what you get.", "Not worth the price, competitors do more for less.", "Too expensive given the build quality."]),
 "calls": (["Microphone is clear, people hear me well on calls.", "Call quality is great even outdoors."],
           ["Callers say I sound like I'm in a tunnel.", "Mic picks up wind noise and my voice is distorted on calls."]),
 "sound": (["Sound is rich with punchy bass and clear vocals.", "Excellent sound quality, very balanced."],
           ["Bass is muddy and the treble is harsh.", "Sound is flat and lifeless compared to my old pair."]),
 "build": (["Build quality feels premium and sturdy.", "Case is solid and the hinge feels durable."],
           ["The case hinge broke within a month.", "Left earbud stopped charging after light use, feels flimsy."]),
}
FILL = ["Using them daily.", "Bought for commuting.", "Ordered for my son.", "Second purchase.", "Delivery was quick.", "", "", ""]
rows, i = [], 1
for a, (pos, neg) in ASPECTS.items():
    for _ in range(25):
        for good in (True, False):
            main = random.choice(pos if good else neg)
            parts = [main, random.choice(FILL)]
            other = random.choice(list(ASPECTS))
            if other != a and random.random() < 0.3:
                op = random.choice(ASPECTS[other][0 if good else 1])
                parts.append(op)
            random.shuffle(parts)
            rating = random.choice([4, 5, 5]) if good else random.choice([1, 2, 2, 3])
            rows.append({"id": f"R{i}", "rating": rating, "title": "",
                         "text": " ".join(p for p in parts if p), "date": f"2026-{random.randint(1,10):02d}-{random.randint(1,28):02d}", "aspect": a})
            i += 1
random.shuffle(rows)
for n, r in enumerate(rows, 1):
    r["id"] = f"R{n}"
with open("data/demo_reviews_synthetic.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print(len(rows), "reviews")
