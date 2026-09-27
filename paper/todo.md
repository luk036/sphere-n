# TODO

## References still to acquire

The following cited works could not be downloaded from any legitimate open-access
source (they are behind IEEE / Springer / SIAM paywalls). Download them via
institutional access and save into `refs/` (see naming convention at the bottom).

- [ ] **`halton1960`** — J. H. Halton, "On the efficiency of certain quasi-random
  sequences of points in evaluating multi-dimensional integrals,"
  *Numerische Mathematik*, vol. 2, pp. 84–90, 1960.
  - DOI: <https://doi.org/10.1007/BF01386213>
  - Landing: <https://link.springer.com/article/10.1007/BF01386213>
  - PDF: <https://link.springer.com/content/pdf/10.1007/BF01386213.pdf>

- [ ] **`cui1997equidistribution`** — J. Cui and W. Freeden, "Equidistribution on
  the Sphere," *SIAM J. Sci. Comput.*, vol. 18, no. 2, pp. 595–609, 1997.
  - DOI: <https://doi.org/10.1137/S1064827595281344>
  - Landing: <https://epubs.siam.org/doi/10.1137/S1064827595281344>
  - PDF: <https://epubs.siam.org/doi/pdf/10.1137/S1064827595281344>
  - Preprint record (metadata only, no full text):
    <https://kluedo.ub.rptu.de/frontdoor/index/index/year/2000/docId/574>

- [ ] **`mitchell2008sampling`** — J. C. Mitchell, "Sampling Rotation Groups by
  Successive Orthogonal Images," *SIAM J. Sci. Comput.*, vol. 30, no. 1,
  pp. 525–547, 2008.
  - DOI: <https://doi.org/10.1137/030601879>
  - Landing: <https://epubs.siam.org/doi/10.1137/030601879>
  - PDF: <https://epubs.siam.org/doi/pdf/10.1137/030601879>
  - Author page: <https://people.math.wisc.edu/~jcmitchell/>

- [ ] **`utkovski2006construction`** — Z. Utkovski and J. Lindner, "On the
  Construction of Non-coherent Space Time Codes from High-dimensional Spherical
  Codes," *IEEE ISSSTA 2006*, pp. 327–331.
  - DOI: <https://doi.org/10.1109/ISSSTA.2006.311788>
  - IEEE Xplore search:
    <https://ieeexplore.ieee.org/search/searchresult.jsp?queryText=non-coherent%20space%20time%20codes%20high-dimensional%20spherical%20codes>
  - Ulm repository (related works): <https://oparu.uni-ulm.de/>

- [ ] **`fishman1996`** — G. S. Fishman, *Monte Carlo: Concepts, Algorithms, and
  Applications*, Springer Series in Operations Research. New York: Springer, 1996.
  - DOI: <https://doi.org/10.1007/978-1-4757-2553-7>
  - Landing: <https://link.springer.com/book/10.1007/978-1-4757-2553-7>
  - ISBN 978-0-387-94527-9 (chapters downloadable individually)

### Naming convention
`refs/<bibkey>_<short-title>.pdf`, e.g. `refs/halton1960_quasi_random_sequences.pdf`.

### Already in `refs/`
`vandercorput1935`, `sobol1967`, `saff1997`, `marsaglia1972`, `strohmer2003`,
`love2003`, `brauchart2012`, `rehman2010`, `mandic2011`, `wong1997`,
`yershova2010`, and `shoemake1992` (inside the full *Graphics Gems III* book PDF).

## Paper build

- [ ] Commit the untracked `paper/ieee.csl` — it is referenced by `paper/Makefile`
      (`--csl=ieee.csl`) and a fresh clone will not build without it.
- [ ] Confirm `make paper` still passes (exit 0, 0 unresolved `??`) after the
      `refs/` additions and the latest `n-sphere.md` edits.

## Optional follow-ups

- [ ] Extract Shoemake's chapter (Graphics Gems III, pp. 124–132) from the full
      book PDF in `refs/` into a standalone file.
- [ ] Trim the 44 MB `vandercorput1935_verteilungsfunktionen_I.pdf` (full 582-page
      KNAW proceedings volume) down to the article (pp. 813–821).
