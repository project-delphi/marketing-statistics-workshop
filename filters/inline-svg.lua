-- Inlines an SVG file into the HTML page, so the figure can read the page's CSS custom
-- properties (--ms-fig-*, set in custom.scss) and follow the light/dark toggle. An <img>
-- cannot: the page's styles do not reach inside an image.
--
--   ::: {.inline-svg src="/images/integration.svg"}
--   Optional text after the figure (a caption, a reading guide).
--   :::
--
-- The div keeps its other classes and content; the SVG becomes its first child. `src`
-- starting with "/" is relative to the project root, otherwise to the page's folder.
-- In a non-HTML format the file is placed as an ordinary image instead.
--
-- The SVG file must carry its own <title>/<desc> (role="img", aria-labelledby) and
-- namespace its ids and classes, because once inlined they share the page.

local function read_file(path)
  local fh = io.open(path, "r")
  if not fh then
    return nil
  end
  local text = fh:read("a")
  fh:close()
  return text
end

local function resolve(src)
  if src:sub(1, 1) == "/" then
    local root = (quarto.project and quarto.project.directory) or "."
    return pandoc.path.join({ root, src:sub(2) })
  end
  local input = quarto.doc.input_file or "."
  return pandoc.path.join({ pandoc.path.directory(input), src })
end

function Div(div)
  if not div.classes:includes("inline-svg") then
    return nil
  end
  local src = div.attributes["src"]
  if not src or src == "" then
    return nil
  end
  div.attributes["src"] = nil
  if not quarto.doc.is_format("html") then
    div.content:insert(1, pandoc.Para({ pandoc.Image({}, src) }))
    return div
  end
  local path = resolve(src)
  local svg = read_file(path)
  if not svg then
    quarto.log.warning("inline-svg: cannot read " .. path)
    return div
  end
  -- Drop an XML declaration or doctype; inline SVG in HTML starts at <svg>.
  svg = svg:gsub("^%s*<%?xml.-%?>%s*", ""):gsub("^%s*<!DOCTYPE.->%s*", "")
  div.content:insert(1, pandoc.RawBlock("html", svg))
  return div
end
