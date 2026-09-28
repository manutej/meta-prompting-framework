package main

import (
	"bytes"
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/alecthomas/chroma/v2"
	"github.com/alecthomas/chroma/v2/formatters"
	"github.com/alecthomas/chroma/v2/lexers"
	"github.com/alecthomas/chroma/v2/styles"
	"github.com/mattn/go-runewidth"
)

var (
	diffLexer   = lexers.Get("diff")
	statLineRe  = regexp.MustCompile(`^(.* \| +\d+ )([+\-]*)$`)
	tabReplacer = strings.NewReplacer("\t", "    ", "\r", "")
)

func renderContent(d diffMsg, width int) string {
	if width < 4 {
		width = 4
	}
	if d.err != nil {
		lines := []string{stError.Render(truncPlain("git: "+d.err.Error(), width))}
		if strings.TrimSpace(d.text) != "" {
			lines = append(lines, "", stMuted.Render(truncPlain(strings.TrimSpace(d.text), width)))
		}
		return strings.Join(lines, "\n")
	}
	text := tabReplacer.Replace(d.text)
	if strings.TrimSpace(text) == "" {
		return stMuted.Render(truncPlain("(no differences)", width))
	}
	switch d.kind {
	case "log":
		return renderLog(text, width)
	case "file":
		return renderFile(d.path, text, width)
	default:
		return renderDiff(text, width)
	}
}

func renderDiff(text string, width int) string {
	it, err := diffLexer.Tokenise(nil, text)
	if err != nil {
		return plainLines(text, width)
	}
	var out []string
	var cur strings.Builder
	var curType chroma.TokenType
	lineStart := true
	for tok := it(); tok != chroma.EOF; tok = it() {
		parts := strings.Split(tok.Value, "\n")
		for i, p := range parts {
			if lineStart {
				curType = tok.Type
				lineStart = false
			}
			cur.WriteString(p)
			if i < len(parts)-1 {
				out = append(out, styleDiffLine(curType, cur.String(), width))
				cur.Reset()
				lineStart = true
			}
		}
	}
	if cur.Len() > 0 {
		out = append(out, styleDiffLine(curType, cur.String(), width))
	}
	return strings.Join(out, "\n")
}

func styleDiffLine(t chroma.TokenType, s string, width int) string {
	s = truncPlain(s, width)
	switch {
	case strings.HasPrefix(s, "--- ") || strings.HasPrefix(s, "+++ ") || strings.HasPrefix(s, "diff --git"):
		return stFileHeader.Render(s)
	case strings.HasPrefix(s, "commit "):
		return stGoldBold.Render(s)
	case strings.HasPrefix(s, "Author:") || strings.HasPrefix(s, "Date:") || strings.HasPrefix(s, "Merge:"):
		return stMuted.Render(s)
	}
	switch t {
	case chroma.GenericInserted:
		return stAdded.Render(s)
	case chroma.GenericDeleted:
		return stRemoved.Render(s)
	case chroma.GenericSubheading:
		return stHunk.Render(s)
	case chroma.GenericHeading:
		return stFileHeader.Render(s)
	case chroma.GenericStrong:
		return stWarn.Render(s)
	}
	if m := statLineRe.FindStringSubmatch(s); m != nil {
		plus := strings.Count(m[2], "+")
		minus := strings.Count(m[2], "-")
		return stText.Render(m[1]) + stAdded.Render(strings.Repeat("+", plus)) + stRemoved.Render(strings.Repeat("-", minus))
	}
	return stText.Render(s)
}

func renderLog(text string, width int) string {
	var out []string
	for _, line := range strings.Split(strings.TrimRight(text, "\n"), "\n") {
		f := strings.Split(line, fieldSep)
		if len(f) < 4 {
			continue
		}
		head := "● " + f[0] + " "
		subject := truncPlain(f[3], width-runewidth.StringWidth(head))
		out = append(out,
			stGold.Render("● ")+stGoldBold.Render(f[0])+" "+stText.Render(subject),
			stMuted.Render(truncPlain("  "+shortAge(f[1])+" · "+f[2], width)),
		)
	}
	if len(out) == 0 {
		return stMuted.Render("(no commits)")
	}
	return strings.Join(out, "\n")
}

func renderFile(path, text string, width int) string {
	if strings.HasPrefix(text, "binary file (") {
		return stMuted.Render(truncPlain(text, width))
	}
	text = strings.TrimRight(text, "\n")
	lexer := lexers.Match(filepath.Base(path))
	if lexer == nil {
		lexer = lexers.Fallback
	}
	lexer = chroma.Coalesce(lexer)
	formatter := formatters.Get("terminal256")
	style := styles.Get("monokai")
	var buf bytes.Buffer
	highlighted := text
	if it, err := lexer.Tokenise(nil, text); err == nil {
		if err := formatter.Format(&buf, style, it); err == nil {
			highlighted = buf.String()
		}
	}
	lines := strings.Split(strings.TrimRight(highlighted, "\n"), "\n")
	gutter := len(fmt.Sprint(len(lines))) + 3
	if gutter > width-2 {
		return plainLines(text, width)
	}
	out := make([]string, 0, len(lines))
	for i, l := range lines {
		num := stMuted.Render(fmt.Sprintf("%*d │ ", gutter-3, i+1))
		out = append(out, num+ansiSlice(l, 0, width-gutter))
	}
	return strings.Join(out, "\n")
}

func plainLines(text string, width int) string {
	lines := strings.Split(strings.TrimRight(text, "\n"), "\n")
	for i, l := range lines {
		lines[i] = stText.Render(truncPlain(l, width))
	}
	return strings.Join(lines, "\n")
}
