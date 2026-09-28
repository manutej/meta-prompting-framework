package main

type hint struct {
	key  string
	desc string
}

func hintsFor(p paneID) []hint {
	switch p {
	case paneStatus:
		return []hint{
			{"space", "stage"}, {"a/A", "all/none"}, {"d", "discard"}, {"s", "stash"},
			{"c", "commit"}, {"enter", "view"}, {"/", "filter"}, {"?", "help"},
		}
	case paneBranches:
		return []hint{
			{"enter", "checkout"}, {"n", "new branch"}, {"c", "commit"},
			{"/", "filter"}, {"r", "refresh"}, {"?", "help"},
		}
	case paneCommits:
		return []hint{
			{"enter", "view"}, {"y", "copy sha"}, {"c", "commit"},
			{"/", "filter"}, {"r", "refresh"}, {"?", "help"},
		}
	case paneStash:
		return []hint{
			{"enter", "pop"}, {"c", "commit"}, {"/", "filter"}, {"r", "refresh"}, {"?", "help"},
		}
	default:
		return []hint{
			{"j/k", "scroll"}, {"ctrl+d/u", "page"}, {"g/G", "top/end"},
			{"h", "back"}, {"c", "commit"}, {"?", "help"},
		}
	}
}

var helpNavigation = []hint{
	{"1-4", "focus a pane"},
	{"tab / S-tab", "next / prev pane"},
	{"h / l", "lists ↔ diff pane"},
	{"j / k", "move · scroll diff"},
	{"ctrl+d / u", "half page in diff"},
	{"g / G", "top / bottom"},
	{"/", "fuzzy filter · esc clears"},
	{"r / F5", "refresh everything"},
	{"mouse", "click selects · wheel"},
	{"?", "toggle this help"},
	{"q", "quit"},
}

var helpActions = []hint{
	{"space", "stage / unstage file"},
	{"a / A", "stage all / unstage all"},
	{"d", "discard file (confirm)"},
	{"s", "stash all (confirm)"},
	{"c", "commit staged changes"},
	{"enter", "view · checkout · pop"},
	{"n", "new branch + checkout"},
	{"y", "copy commit SHA"},
}
