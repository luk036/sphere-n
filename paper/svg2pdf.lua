-- svg2pdf.lua -- build-time image remapping for the LaTeX/PDF pipeline.
--
-- Why this exists (md-to-latex skill, rule 7):
--   pdflatex/xelatex cannot embed SVG. rsvg-convert (librsvg) is on PATH, so
--   `make svg` pre-renders every referenced *.svg to a sibling *.pdf. This
--   filter points pandoc at the *.pdf twin without touching the Markdown.
--
-- The geometric figures (vdcorput/circle/halton/halton3d/sphere/disk) carry
-- huge natural page sizes (2048pt). With no width attribute pandoc would emit
-- an unconstrained \includegraphics and the figure would overflow the text
-- block, so we default such images to \linewidth.

local function is_svg(src)
  return src ~= nil and src:match("%.svg$") ~= nil
end

function Image(el)
  if is_svg(el.src) then
    el.src = el.src:gsub("%.svg$", ".pdf")
    if el.attributes.width == nil then
      el.attributes.width = "100%"
    end
  end
  return el
end
