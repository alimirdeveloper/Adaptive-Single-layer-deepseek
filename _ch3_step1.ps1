$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# Resolve paths without hardcoding Persian literals in this file
$dir = 'C:\Users\Asus\Desktop\uni\proposal git'

$srcFile = Get-ChildItem -LiteralPath $dir -Filter '*.docx' |
           Where-Object { $_.Name -notlike '*_*' } |
           Sort-Object LastWriteTime -Descending |
           Select-Object -First 1

# The genuine original is the 975309-byte file
$srcFile = Get-ChildItem -LiteralPath $dir -Filter '*.docx' |
           Where-Object { $_.Length -eq 975309 } | Select-Object -First 1

Write-Output ("SRC = " + $srcFile.FullName)

# Build the copy path with a suffix appended programmatically
$base = [System.IO.Path]::GetFileNameWithoutExtension($srcFile.Name)
$newName = $base + [char]0x005F + [char]0x0641 + [char]0x0635 + [char]0x0644 + [char]0x06F3 + '.docx'
$DST = Join-Path $dir $newName
Write-Output ("DST = " + $DST)

Copy-Item -LiteralPath $srcFile.FullName -Destination $DST -Force

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0

$script:doc = $word.Documents.Open($DST, $false, $false)
Write-Output ("Opened. Paragraphs = " + $script:doc.Paragraphs.Count)

# Locate Chapter 3 boundaries using codepoint-safe matching
$startIdx = -1
$endIdx = -1
$i = 0
foreach ($p in $script:doc.Paragraphs) {
    $i++
    $styleName = $p.Style.NameLocal
    $txt = $p.Range.Text.Trim()
    if ($styleName -eq 'Heading 1') {
        # "فصل سوم"  = U+0641 U+0635 U+0644 ' ' U+0633 U+0648 U+0645
        if ($txt -match ([char]0x0641 + [char]0x0635 + [char]0x0644 + '\s*' + [char]0x0633 + [char]0x0648 + [char]0x0645)) {
            if ($startIdx -lt 0) { $startIdx = $i }
        }
        # "فصل چهارم"
        elseif ($txt -match ([char]0x0641 + [char]0x0635 + [char]0x0644 + '\s*' + [char]0x0686 + [char]0x0647 + [char]0x0627 + [char]0x0631 + [char]0x0645)) {
            if ($startIdx -gt 0) { $endIdx = $i; break }
        }
    }
}
Write-Output ("Ch3 start paragraph = " + $startIdx + " ; Ch4 start paragraph = " + $endIdx)
if ($startIdx -lt 0 -or $endIdx -lt 0) { throw "Could not locate chapter boundaries" }

$rStart = $script:doc.Paragraphs.Item($startIdx).Range.Start
$rEnd   = $script:doc.Paragraphs.Item($endIdx).Range.Start

$probe = $script:doc.Range($rStart, $rEnd)
Write-Output ("Tables inside old Ch3 = " + $probe.Tables.Count + " ; InlineShapes = " + $probe.InlineShapes.Count)

$probe.Delete() | Out-Null
Write-Output ("Old Ch3 deleted. Paragraphs now = " + $script:doc.Paragraphs.Count)

$script:doc.Save()
$script:doc.Close($true)
$word.Quit()
Write-Output 'STEP1 DONE'
