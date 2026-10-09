-- Wraps every table of an HTML page in <div class="table-scroll">, so a table wider than
-- the text column scrolls inside its own box (custom.scss) instead of pushing the page
-- sideways at phone width. Generated includes (timetables, the notebooks list, readiness)
-- and hand-written tables get the same treatment without any markup of their own.
--
-- Cross-referenced tables (an id starting with "tbl-") are left alone, so Quarto still
-- sees them as figures-like floats.

function Table(tbl)
  if not quarto.doc.is_format("html") then
    return nil
  end
  if tbl.identifier and tbl.identifier:match("^tbl%-") then
    return nil
  end
  return pandoc.Div({ tbl }, pandoc.Attr("", { "table-scroll" }))
end
