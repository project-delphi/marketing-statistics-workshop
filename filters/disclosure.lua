-- Gives collapsed callouts (`collapse="true"`) a native, keyboard-operable
-- disclosure. It never changes the text of a page. Adapted from
-- project-delphi/nlp-llms (MIT).
--
-- Quarto (checked on 1.10.19) renders a collapsed callout as a <div> header with
-- a Bootstrap collapse toggle: it has no tabindex and no role, so it cannot be
-- reached from the keyboard, and its aria-label ("Toggle callout") hides the title. This
-- filter writes the callout as a <details> element with a <summary> header
-- instead: a Tab stop, opened with Enter or Space, announced with its title and
-- its expanded state. The outer <div> keeps Quarto's callout classes, so the
-- frame, colors and icons look as before (custom.scss styles the <summary>).
--
-- A link to something inside a closed callout opens it, and printing opens them
-- all (the script below, added to pages that have at least one).
--
-- Runs before Quarto turns callout divs into its own nodes (`at: pre-ast` in
-- _quarto.yml), after filters/pedagogy.lua, so a callout is still a plain Div
-- here and keeps the attributes the earlier filters set (`data-optional`). The
-- output must not be a pandoc Div with a callout-* class, or Quarto would render
-- it again. After a Quarto upgrade, check that no `data-bs-toggle="collapse"` is
-- left inside a callout in the rendered site (grep _site for it).

local TYPES = { note = true, tip = true, warning = true, important = true, caution = true }
-- The document's callout defaults (`callout-appearance`, `callout-icon`), read from its
-- metadata before any callout is written, so collapsed callouts follow them as
-- Quarto's own callouts do.
local defaults = { appearance = "default", icon = true }
-- Callout options that Quarto reads; everything else is copied to the HTML.
local OPTIONS = { title = true, collapse = true, appearance = true, icon = true }

local SCRIPT = [[
<script>
// Collapsed callouts are details elements (filters/disclosure.lua). A link to something
// inside a closed one opens it; printing opens them all, then restores them.
(function () {
  function openTarget() {
    var id;
    try { id = decodeURIComponent(location.hash.slice(1)); } catch (e) { return; }
    var target = id && document.getElementById(id);
    if (!target) return;
    var opened = false;
    // The callout itself (its id is on the frame around the details), or anything inside one.
    var own = target.querySelector(":scope > details.callout-details");
    if (own && !own.open) { own.open = true; opened = true; }
    for (var el = target; el; el = el.parentElement) {
      if (el.tagName === "DETAILS" && !el.open) { el.open = true; opened = true; }
    }
    if (opened) target.scrollIntoView();
  }
  addEventListener("hashchange", openTarget);
  addEventListener("DOMContentLoaded", openTarget);
  var closed = [];
  addEventListener("beforeprint", function () {
    closed = Array.from(document.querySelectorAll("details:not([open])"));
    closed.forEach(function (d) { d.open = true; });
  });
  addEventListener("afterprint", function () {
    closed.forEach(function (d) { d.open = false; });
  });
})();
</script>
]]

local added_script = false

local function starts_with(text, prefix)
  return text:sub(1, #prefix) == prefix
end

local function escape(s)
  return (s:gsub("&", "&amp;"):gsub('"', "&quot;"):gsub("<", "&lt;"))
end

local function callout_type(div)
  for _, class in ipairs(div.classes) do
    local t = class:match("^callout%-(%a+)$")
    if t and TYPES[t] then
      return t
    end
  end
end

-- The title is the `title` attribute (markdown) or, in the older form, the
-- heading that opens the callout; the heading is then not part of the body.
local function title_and_body(div)
  local t = div.attributes["title"]
  if t and t ~= "" then
    local first = pandoc.read(t, "markdown").blocks[1]
    if first and (first.t == "Para" or first.t == "Plain") then
      return first.content, div.content
    end
    -- A title that markdown reads as a list ("1. Why `softmax`?"): keep its marker as
    -- text and the markup of the rest. Anything else stays literal text.
    local marker = t:match("^%s*(%d+[.)])%s") or t:match("^%s*([-*+])%s")
    if marker and first and (first.t == "OrderedList" or first.t == "BulletList") then
      local inner = pandoc.utils.blocks_to_inlines(first.content[1])
      return pandoc.Inlines({ pandoc.Str(marker), pandoc.Space() }) .. inner, div.content
    end
    return pandoc.Inlines(t), div.content
  end
  local first = div.content[1]
  if first and first.t == "Header" then
    local rest = pandoc.Blocks({})
    for i = 2, #div.content do
      rest:insert(div.content[i])
    end
    return first.content, rest
  end
  return pandoc.Inlines({}), div.content
end

local function read_defaults(meta)
  if meta["callout-appearance"] ~= nil then
    defaults.appearance = pandoc.utils.stringify(meta["callout-appearance"])
  end
  local icon = meta["callout-icon"]
  if icon ~= nil then
    defaults.icon = not (icon == false or pandoc.utils.stringify(icon) == "false")
  end
end

local function Div(div)
  local ctype = callout_type(div)
  if not ctype or div.attributes["collapse"] ~= "true" or not FORMAT:match("html") then
    return nil
  end
  local title, body = title_and_body(div)
  if #title == 0 then
    title = pandoc.Inlines({ pandoc.Str(ctype:sub(1, 1):upper() .. ctype:sub(2)) })
  end
  -- Quarto's "minimal" is "simple" without an icon.
  local appearance = div.attributes["appearance"] or defaults.appearance
  local minimal = appearance == "minimal"
  local classes = {
    "callout",
    "callout-style-" .. (minimal and "simple" or appearance),
    "callout-" .. ctype,
    "callout-titled",
    "callout-disclosure",
  }
  local icon = div.attributes["icon"]
  if minimal or icon == "false" or (icon == nil and not defaults.icon) then
    table.insert(classes, "no-icon")
  end
  for _, class in ipairs(div.classes) do
    if not starts_with(class, "callout") then
      table.insert(classes, class)
    end
  end
  local attrs = ' class="' .. table.concat(classes, " ") .. '"'
  if div.identifier ~= "" then
    attrs = attrs .. ' id="' .. escape(div.identifier) .. '"'
  end
  for key, value in pairs(div.attributes) do
    if not OPTIONS[key] then
      attrs = attrs .. " " .. key .. '="' .. escape(value) .. '"'
    end
  end
  if not added_script then
    quarto.doc.include_text("after-body", SCRIPT)
    added_script = true
  end
  return {
    pandoc.RawBlock("html", "<div" .. attrs .. ">\n"
      .. '<details class="callout-details">\n'
      -- Only phrasing content inside <summary>; the row span works around
      -- Safari, which does not lay out a <summary> as a flex container.
      .. '<summary class="callout-header"><span class="callout-header-row">'
      .. '<span class="callout-icon-container" aria-hidden="true"><i class="callout-icon"></i></span>'
      .. '<span class="callout-title-container">'),
    pandoc.Plain(title),
    pandoc.RawBlock("html", '</span><span class="callout-btn-toggle" aria-hidden="true">'
      .. '<i class="callout-toggle"></i></span></span></summary>'),
    pandoc.Div(body, pandoc.Attr("", { "callout-body-container", "callout-body" })),
    pandoc.RawBlock("html", "</details>\n</div>"),
  }
end

-- Two passes: the metadata first, then the callouts.
return { { Meta = read_defaults }, { Div = Div } }
