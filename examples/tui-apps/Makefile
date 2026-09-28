DEMO  := nexus-command gitscope docscope
BASIC := progress-timer file-browser system-monitor
APPS  := $(DEMO) $(BASIC)

.PHONY: build demo test vet smoke clean $(APPS)

build: $(APPS)

demo: $(DEMO)

$(APPS):
	cd $@ && go build -o $@ .

vet:
	@for a in $(APPS); do (cd $$a && go vet ./...) || exit 1; done

test: vet
	@for a in $(APPS); do echo "== $$a" && (cd $$a && go test ./...) || exit 1; done

smoke: build
	python3 scripts/pty_smoke.py
	@for a in $(DEMO); do echo "== $$a" && python3 $$a/smoke.py || exit 1; done

clean:
	@for a in $(APPS); do rm -f $$a/$$a; done
