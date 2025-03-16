import spacy

# download the En model firstly by：python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# This is only a test method for color prompt, like "a red car and a blue bench"
def split_prompt(prompt):
    # Step 1: parse
    doc = nlp(prompt)
    chunks = list(doc.noun_chunks)

    # If only one chunk and there's "and", try splitting on " and "
    if len(chunks) < 2 and " and " in prompt:
        sub_prompts = [seg.strip() for seg in prompt.split(" and ")]
        new_chunks = []
        for sub in sub_prompts:
            sub_doc = nlp(sub)
            sub_noun_chunks = list(sub_doc.noun_chunks)
            if not sub_noun_chunks:
                new_chunks.append(sub_doc[:])
            else:
                new_chunks.extend(sub_noun_chunks)
        chunks = new_chunks

    if len(chunks) < 2:
        sps = [prompt]
        nps = [prompt]
        return (sps, nps)

    # Step 2: split on "of" within each chunk
    final_chunks = []
    for chunk in chunks:
        chunk_tokens = list(chunk)
        of_prep_token = None
        for token in chunk_tokens:
            if token.lemma_ == "of" and token.dep_ == "prep":
                of_prep_token = token
                break

        if of_prep_token:
            of_subtree = list(of_prep_token.subtree)
            chunk1_tokens = [t for t in chunk_tokens if t not in of_subtree]
            pobj_tokens = [t for t in of_prep_token.children if t.dep_ == "pobj"]
            if pobj_tokens:
                pobj_root = pobj_tokens[0]
                chunk2_tokens = list(pobj_root.subtree)
            else:
                chunk2_tokens = []

            if chunk1_tokens and chunk2_tokens:
                doc_obj = chunk1_tokens[0].doc
                chunk1_span = doc_obj[chunk1_tokens[0].i : chunk1_tokens[-1].i + 1]
                chunk2_span = doc_obj[chunk2_tokens[0].i : chunk2_tokens[-1].i + 1]
                final_chunks.append(chunk1_span)
                final_chunks.append(chunk2_span)
                continue
            else:
                final_chunks.append(chunk)
        else:
            final_chunks.append(chunk)

    if len(final_chunks) < 2:
        sps = [prompt]
        nps = [prompt]
        return (sps, nps)

    def process_chunk(ch):
        full = ch.text
        det = ""
        for token in ch:
            if token.dep_ == "det":
                det = token.text
                break
        head = ch.root.text
        stripped = (det + " " if det else "") + head
        return (full, stripped)

    processed = [process_chunk(ch) for ch in final_chunks]

    variants = []
    for i in range(len(processed)):
        parts = []
        for j, (full, stripped) in enumerate(processed):
            if j == i:
                parts.append(full)
            else:
                parts.append(stripped)
        variants.append(" and ".join(parts))

    # simple_version = " and ".join(ch.root.text for ch in final_chunks)
    simple_version = " and ".join(stripped for full, stripped in processed)
    subjects = [ch.root.text for ch in final_chunks]

    sps = [simple_version]
    sps.extend(variants)
    nps = subjects

    return (sps, nps)
