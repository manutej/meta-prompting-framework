package main

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const maxTreeDepth = 6

var skipDirs = map[string]bool{"node_modules": true, ".git": true, "vendor": true}

type node struct {
	name     string
	path     string // absolute
	rel      string // relative to the root, slash-separated
	isDir    bool
	depth    int
	expanded bool
	children []*node
}

// buildTree walks root for *.md files (max depth 6, skipping node_modules,
// .git and vendor) and returns the directory tree plus all files in tree order.
func buildTree(root string) (*node, []*node, error) {
	if _, err := os.Stat(root); err != nil {
		return nil, nil, err
	}
	rootNode := &node{name: filepath.Base(root), path: root, isDir: true, expanded: true}
	var rels []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			if p == root {
				return err
			}
			if d != nil && d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if p == root {
			return nil
		}
		rel, rerr := filepath.Rel(root, p)
		if rerr != nil {
			return nil
		}
		depth := strings.Count(rel, string(filepath.Separator)) + 1
		if d.IsDir() {
			if skipDirs[d.Name()] || depth >= maxTreeDepth {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.EqualFold(filepath.Ext(d.Name()), ".md") {
			rels = append(rels, rel)
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}
	for _, rel := range rels {
		insert(rootNode, root, rel)
	}
	sortTree(rootNode)
	var files []*node
	collectFiles(rootNode, &files)
	return rootNode, files, nil
}

func insert(root *node, rootPath, rel string) {
	parts := strings.Split(rel, string(filepath.Separator))
	cur := root
	for i, part := range parts {
		isLast := i == len(parts)-1
		var next *node
		for _, c := range cur.children {
			if c.name == part && c.isDir == !isLast {
				next = c
				break
			}
		}
		if next == nil {
			next = &node{
				name:     part,
				path:     filepath.Join(rootPath, filepath.Join(parts[:i+1]...)),
				rel:      filepath.ToSlash(filepath.Join(parts[:i+1]...)),
				isDir:    !isLast,
				depth:    i + 1,
				expanded: true,
			}
			cur.children = append(cur.children, next)
		}
		cur = next
	}
}

func sortTree(n *node) {
	sort.SliceStable(n.children, func(i, j int) bool {
		a, b := n.children[i], n.children[j]
		if a.isDir != b.isDir {
			return a.isDir
		}
		return strings.ToLower(a.name) < strings.ToLower(b.name)
	})
	for _, c := range n.children {
		if c.isDir {
			sortTree(c)
		}
	}
}

func collectFiles(n *node, out *[]*node) {
	for _, c := range n.children {
		if c.isDir {
			collectFiles(c, out)
		} else {
			*out = append(*out, c)
		}
	}
}

// visibleRows flattens the tree honouring each directory's expanded flag.
func visibleRows(n *node) []*node {
	var rows []*node
	var walk func(*node)
	walk = func(d *node) {
		for _, c := range d.children {
			rows = append(rows, c)
			if c.isDir && c.expanded {
				walk(c)
			}
		}
	}
	walk(n)
	return rows
}

// expandTo opens every ancestor directory of the file so it becomes visible.
func expandTo(root *node, target *node) {
	var walk func(*node) bool
	walk = func(d *node) bool {
		for _, c := range d.children {
			if c == target {
				return true
			}
			if c.isDir && walk(c) {
				c.expanded = true
				return true
			}
		}
		return false
	}
	walk(root)
}
