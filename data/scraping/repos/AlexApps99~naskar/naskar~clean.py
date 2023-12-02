def clean(gpt, prune=True, log_pruned=True):
  '''
  takes output from OpenAI and cleans and filters for valid sonnets
  '''
  # half-finished sonnets are no good, and if it isn't finished it's too long
  sonnets = [s["text"] for s in gpt["choices"] if s["finish_reason"] == "stop"]
  # strip each line
  sonnets = ["\n".join([l.strip() for l in s.splitlines()]) for s in sonnets]
  # split into stanzas
  sonnets = [[q.strip() for q in s.split("\n\n") if q.strip()] for s in sonnets]
  # split stanzas into lines
  sonnets = [[q.split("\n") for q in s] for s in sonnets]

  def validate_sonnet(sonnet):
    # Too few/many stanzas
    if len(sonnet) < 3 or len(sonnet) > 5: return False
    for stanza in sonnet:
      # Stanza is too long
      if len(stanza) > 4: return False
    # It's probably good enough
    return True

  def regroup_sonnet(sonnet, n):
    i = 0
    while i < len(sonnet) - 1:
      while (i < len(sonnet) - 1) and len(sonnet[i]) + len(sonnet[i+1]) <= n:
        sonnet[i:i+2] = [sonnet[i] + sonnet[i+1]]
      i += 1
    return sonnet

  # merge stanzas together until there would be more than 4 (or 3) lines per stanza
  def try_regroup_sonnet(sonnet):
    for n in [4, 3]:
      new_sonnet = sonnet.copy()
      new_sonnet = regroup_sonnet(new_sonnet, n)
      if validate_sonnet(new_sonnet):
        return new_sonnet
    return None

  def reconstitute(sonnet):
    return "\n\n".join(["\n".join(q) for q in sonnet])

  regrouped_sonnets = [try_regroup_sonnet(s) for s in sonnets]

  if prune:
    def pruned(sonnet):
      v = not bool(sonnet)
      if log_pruned and v: print(sonnet)
      return v

    return [reconstitute(s) for s in regrouped_sonnets if not pruned(s)]
  else:
    return [{
      "text": reconstitute(s if s else sonnets[i]),
      "clean": bool(s),
    } for i, s in enumerate(regrouped_sonnets)]

