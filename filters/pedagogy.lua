-- Adds classes that custom.scss styles. It never changes the text of a page.
-- Adapted from project-delphi/nlp-llms (MIT).
--
--   1. A paragraph that opens with **Objective for this section:** (any bold
--      run starting with "Objective") is wrapped in a div with class
--      `section-objective`: the learning objective before the content.
--   2. A collapsed callout whose title starts with "Optional" gets the
--      attribute `data-optional="true"`: material outside the briefing's time,
--      shown with a dashed frame. The title is the `title` attribute or, in the
--      older form, the heading that opens the callout.
--
-- Runs before Quarto turns callout divs into its own nodes (`at: pre-ast` in
-- _quarto.yml), so a callout is still a plain Div here.

local stringify = pandoc.utils.stringify

local function starts_with(text, prefix)
  return text:sub(1, #prefix) == prefix
end

local function is_callout(div)
  for _, class in ipairs(div.classes) do
    if starts_with(class, "callout-") then
      return true
    end
  end
  return false
end

local function callout_title(div)
  if div.attributes["title"] then
    return div.attributes["title"]
  end
  local first = div.content[1]
  if first and first.t == "Header" then
    return stringify(first)
  end
  return ""
end

function Para(para)
  local first = para.content[1]
  if first and first.t == "Strong" and starts_with(stringify(first), "Objective") then
    return pandoc.Div({ para }, pandoc.Attr("", { "section-objective" }))
  end
end

function Div(div)
  if is_callout(div) and div.attributes["collapse"] == "true"
      and starts_with(callout_title(div), "Optional") then
    -- Quarto's callout renderer keeps attributes but drops extra classes.
    div.attributes["data-optional"] = "true"
    return div
  end
end
