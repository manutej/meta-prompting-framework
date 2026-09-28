DEMO  := nexus-command gitscope docscope alembic
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
	python3 scripts/smoke.py $(APPS)

clean:
	@for a in $(APPS); do rm -f $$a/$$a; done
