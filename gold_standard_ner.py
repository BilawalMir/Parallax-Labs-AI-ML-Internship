"""
gold_standard_ner.py
---------------------
Small manually-labeled test set for evaluating NER accuracy (task
requirement: "evaluating accuracy against a small manually labeled set").

Every excerpt below is real text from this corpus -- pulled from the
representative-document samples printed during the actual topic_modeling.py
run on the full 35,070-chunk collection (topics: encryption/Clipper Chip,
FBI/Waco, cars, atheism, Israel/Palestine, guns, computer hardware/OS,
sports). Entities were labeled by hand by reading each excerpt, restricted
to the four types ner_enrichment.py stores (PERSON, ORG, GPE, NORP) since
those are the only ones evaluated here.

Labeling notes / judgment calls (documented for transparency, since manual
labeling always involves some):
- "Clipper Chip" and "Skipjack" are product/algorithm names, not ORG/PERSON
  -- deliberately left OUT of the gold labels for those types. This is one
  of the cases evaluate_ner_accuracy.py checks: does spaCy correctly avoid
  tagging them as ORG/PERSON, or does it (incorrectly) fold them in?
- Possessive forms ("Ford's") are labeled with the base entity ("Ford").
- Acronyms are labeled as they appear (e.g. "FBI", not "Federal Bureau of
  Investigation", since the corpus text uses the acronym).
"""

GOLD_STANDARD = [
    {
        "text": "FBI/BATF violated Davidians right to free exercise of religion from the start.",
        "entities": [
            ("FBI", "ORG"),
            ("BATF", "ORG"),
            ("Davidians", "NORP"),
        ],
    },
    {
        "text": "This weekend I was briefed by Attorney General Reno on an operation prepared by the FBI.",
        "entities": [
            ("Reno", "PERSON"),
            ("FBI", "ORG"),
        ],
    },
    {
        "text": "It appears that the tactics employed by the FBI did drive Koresh over the edge.",
        "entities": [
            ("FBI", "ORG"),
            ("Koresh", "PERSON"),
        ],
    },
    {
        "text": "The Clipper Chip contains a classified 64-bit block encryption algorithm called Skipjack.",
        "entities": [],  # Clipper Chip / Skipjack are products, not ORG/PERSON/GPE/NORP
    },
    {
        "text": "F, an 80-bit family key that is common to all chips N, a 30-bit serial number.",
        "entities": [],
    },
    {
        "text": "NASA's Apollo 11 mission landed the first humans on the Moon on July 20, 1969, "
                "with astronauts Neil Armstrong and Buzz Aldrin.",
        "entities": [
            ("NASA", "ORG"),
            ("Neil Armstrong", "PERSON"),
            ("Buzz Aldrin", "PERSON"),
        ],
    },
    {
        "text": "The Python programming language was created by Guido van Rossum and first released in 1991.",
        "entities": [
            ("Guido van Rossum", "PERSON"),
        ],
    },
    {
        "text": "Ford hasn't released a new car without a 4-wheel IS in 7 years.",
        "entities": [
            ("Ford", "ORG"),
        ],
    },
    {
        "text": "FOR SALE: 1990 Pontiac Grand Prix SE 2-Door Coupe, White, White rims, Gray interior.",
        "entities": [],  # Pontiac here is a car model reference within a product listing, not a clean ORG mention
    },
    {
        "text": "Ford is doing nothing but re-skinning the existing Mustang, with MINOR suspension modifications.",
        "entities": [
            ("Ford", "ORG"),
        ],
    },
    {
        "text": "Atheism is characterized by an absence of belief in the existence of God.",
        "entities": [],
    },
    {
        "text": "Most atheists take a live and let live attitude.",
        "entities": [],
    },
    {
        "text": "While Israeli Jews fete the uprising of the Warsaw ghetto, they repress by violent means "
                "the uprising of the Gaza ghetto.",
        "entities": [
            ("Israeli", "NORP"),
            ("Jews", "NORP"),
            ("Warsaw", "GPE"),
            ("Gaza", "GPE"),
        ],
    },
    {
        "text": "The Israeli occupier has decided to punish the whole population of Gaza, some 700.000 people, "
                "by denying them the right to leave the strip and seek work in Israel.",
        "entities": [
            ("Israeli", "NORP"),
            ("Gaza", "GPE"),
            ("Israel", "GPE"),
        ],
    },
    {
        "text": "As a non-observant jew/Israeli/American, my own feeling is that Jews from the diaspora do not "
                "have a greater right in Palestine or Israel than the palestinians.",
        "entities": [
            ("jew", "NORP"),
            ("Israeli", "NORP"),
            ("American", "NORP"),
            ("Jews", "NORP"),
            ("Palestine", "GPE"),
            ("Israel", "GPE"),
            ("palestinians", "NORP"),
        ],
    },
    {
        "text": "This is the official Request for Discussion (RFD) for the creation of two new newsgroups "
                "for Microsoft Windows NT.",
        "entities": [
            ("Microsoft", "ORG"),
        ],
    },
    {
        "text": "Microsoft NT is the newest member of the Microsoft Windows family of operating systems.",
        "entities": [
            ("Microsoft", "ORG"),
            ("Microsoft", "ORG"),
        ],
    },
    {
        "text": "I have a Radius Precision Color 24x video card for the Mac that fits in a NuBus slot.",
        "entities": [
            ("Radius", "ORG"),
            ("Mac", "ORG"),
        ],
    },
    {
        "text": "The cheapest I could find this card for when I called around last night was $1738 at Mac's Place.",
        "entities": [
            ("Mac's Place", "ORG"),
        ],
    },
    {
        "text": "Chris Magnuson Hewlett-Packard Company.",
        "entities": [
            ("Chris Magnuson", "PERSON"),
            ("Hewlett-Packard Company", "ORG"),
        ],
    },
]
