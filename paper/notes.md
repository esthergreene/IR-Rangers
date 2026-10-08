## Notes for paper

  * **Collection**: 34,308 papers (arXiv, August–September 2024), 2,275,574 paragraphs (~66 per paper). Queries: Train 2,000, Study 200, Test 200, each 30/50/20% Easy/Medium/Hard.
  
  * **Cleaning results**: 168.0M → 114.9M tokens (−31.6%), 781k → 660k unique terms, median paragraph length 58 → 40 tokens.

  * **Noise found:**
    * paragraph 1 repeats the abstract, and the abstract field has merged words (imagesor)

    * merged words throughout, including in queries (sustainedvowel in Test 2251)

    * display equations already missing from the text

    * inline math made up about 17% of tokens before cleaning

    * 1,458 paragraphs with an unpaired $
    
    * 8,389 paragraphs empty after cleaning
    
    * 117 paragraphs over 1,000 tokens (author lists, flattened tables)

    * QRELs: few documents judged per query, so P@10 has a low ceiling. _1 paragraphs can be judged relevant.
    
    * Expected failure cases: paraphrased queries (Train 2: "denoising-based generative frameworks"), and a lone "C" dropped by the length filter.