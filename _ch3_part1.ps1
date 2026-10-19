$ErrorActionPreference = 'Stop'

$SRC = 'C:\Users\Asus\Desktop\uni\proposal git\پروپوزال.docx'
$DST = 'C:\Users\Asus\Desktop\uni\proposal git\پروپوزال_فصل۳.docx'

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

$doc = $word.Documents.Open($DST, $false, $false)

# ---------------------------------------------------------------------------
# Locate Chapter 3 boundaries: Heading 1 "فصل سوم" .. just before Heading 1 "فصل چهارم"
# ---------------------------------------------------------------------------
$startIdx = -1
$endIdx = -1
$i = 0
foreach ($p in $script:doc.Paragraphs) {
    $i++
    $styleName = $p.Style.NameLocal
    $txt = $p.Range.Text.Trim()
    if ($styleName -eq 'Heading 1') {
        if ($txt -match 'فصل\s*سوم') { $startIdx = $i }
        elseif ($txt -match 'فصل\s*چهارم' -and $startIdx -gt 0) { $endIdx = $i; break }
    }
}
Write-Output "Ch3 start paragraph = $startIdx ; Ch4 start paragraph = $endIdx"

if ($startIdx -lt 0 -or $endIdx -lt 0) { throw "Could not locate chapter boundaries" }

# Check for real tables / shapes inside the range BEFORE deleting
$tblCount = $script:doc.Range($script:doc.Paragraphs.Item($startIdx).Range.Start,
                              $script:doc.Paragraphs.Item($endIdx).Range.Start).Tables.Count
$shpCount = $script:doc.Range($script:doc.Paragraphs.Item($startIdx).Range.Start,
                              $script:doc.Paragraphs.Item($endIdx).Range.Start).InlineShapes.Count
Write-Output "Tables inside old Ch3 = $tblCount ; InlineShapes = $shpCount"

# Delete old chapter 3
$delRange = $script:doc.Range($script:doc.Paragraphs.Item($startIdx).Range.Start,
                              $script:doc.Paragraphs.Item($endIdx).Range.Start)
$delRange.Delete() | Out-Null
Write-Output "Old Ch3 deleted. Paragraphs now = $($script:doc.Paragraphs.Count)"

# Remember the insertion point
$insertAt = $script:doc.Paragraphs.Item($startIdx).Range.Start
Write-Output "Insertion point = $insertAt"

$script:doc.Save()
$script:doc.Close($true)
$word.Quit()
[System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
Write-Output 'STEP1 DONE'
