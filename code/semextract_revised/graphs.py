import math
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

from .common import write_json


def _label(value, width=35):
    # Graphviz escape sequences in article labels must remain literal.
    return "\n".join(textwrap.fill(line, width=width) for line in str(value).splitlines()).replace("\\", "\\\\")


def _statistics(relation):
    parts = []
    if relation.get("beta") is not None:
        parts.append(f"β = {relation['beta']}")
    p = relation.get("p")
    if p is not None:
        parts.append("p " + (str(p) if str(p).startswith(("<", ">", "=")) else "= " + str(p)))
    return "\n".join(parts)


def _effect_type(relation):
    # Also accept the compact extraction schema, which represents roles directly.
    return relation.get("effect_type") or (
        "indirect" if relation.get("mediator") else
        "moderation" if relation.get("moderator") else "direct")


def build_graph(relations, relation_ids=None, direction="LR", title="Extracted SEM relations"):
    if direction not in {"LR", "TB"}:
        raise ValueError("direction must be LR or TB")
    relation_ids = relation_ids if relation_ids is not None else [f"R{i + 1}" for i in range(len(relations))]
    if len(relation_ids) != len(relations) or len(set(relation_ids)) != len(relation_ids):
        raise ValueError("Provide one unique relation ID per relation")
    grouped = defaultdict(list)
    for rid, relation in zip(relation_ids, relations):
        if not all(isinstance(relation.get(k), str) and relation[k].strip() for k in ("cause", "effect")):
            raise ValueError("Every rendered relation needs cause and effect names")
        effect_type = _effect_type(relation)
        if effect_type not in {"direct", "indirect", "moderation"}:
            raise ValueError(f"Unsupported effect type: {effect_type}")
        mediator = relation.get("mediator")
        chain = mediator if isinstance(mediator, list) else [mediator] if mediator else []
        if any(not isinstance(name, str) or not name.strip() for name in chain):
            raise ValueError("Invalid mediator chain")
        moderator = relation.get("moderator")
        if moderator is not None and (not isinstance(moderator, str) or not moderator.strip()):
            raise ValueError("A moderator must be a nonempty construct name or null")
        if effect_type != "indirect" and chain:
            raise ValueError("A mediator requires an indirect effect")
        if effect_type == "direct" and relation.get("moderator"):
            raise ValueError("A moderator requires a moderation or indirect effect")
        if relation.get("beta") is not None and (isinstance(relation["beta"], bool)
                                                 or not math.isfinite(float(relation["beta"]))):
            raise ValueError("A coefficient must be finite")
        grouped[(relation.get("model_id"), relation.get("group_id"))].append((rid, relation, chain))

    graph = {"directed": True,
             "graphAttributes": {"rankdir": direction, "splines": "polyline", "nodesep": .35,
                                 "ranksep": .65, "pad": .2, "bgcolor": "white", "fontname": "Arial",
                                 "fontsize": 15, "labelloc": "t", "label": _label(title, 90),
                                 "compound": True, "outputorder": "edgesfirst", "dpi": 144},
             "nodeAttributes": {"shape": "box", "style": "rounded,filled", "fillcolor": "#f3f7fa",
                                "fontname": "Arial", "fontsize": 13, "margin": "0.18,0.12", "color": "#577080"},
             "edgeAttributes": {"fontname": "Arial", "fontsize": 11, "arrowsize": .8, "penwidth": 1.5},
             "nodes": [], "edges": [], "subgraphs": []}
    manifest = {"relation_ids": relation_ids, "warnings": [], "nodes": [], "edges": []}
    for gi, ((model, group), records) in enumerate(grouped.items()):
        cluster = {"name": f"cluster_{gi}", "graphAttributes": {
            "label": _label(" | ".join(str(v) for v in (model, group) if v), 80),
            "color": "#c3cdd5", "margin": 22}, "nodes": [], "edges": []}
        nodes = {}

        def node(name, kind="construct", key=None):
            identity = (kind, name if key is None else key)
            if identity in nodes:
                return nodes[identity]
            nid = f"g{gi}_n{len(nodes)}"
            nodes[identity] = nid
            attributes = {"label": _label(name)}
            if kind == "junction":
                attributes.update(label="", shape="point", width=.08, height=.08,
                                  color="#7650a0", fillcolor="#7650a0")
            elif kind == "note":
                attributes.update(shape="note", style="filled", fillcolor="#fff7e6", color="#9b7b3b")
            elif kind == "unknown":
                attributes.update(style="rounded,dashed", fillcolor="white")
            cluster["nodes"].append({"name": nid, "attributes": attributes})
            manifest["nodes"].append({"id": nid, "name": name, "kind": kind, "model_id": model, "group_id": group})
            return nid

        def edge(start, end, rid, kind="direct", label="", arrow=True):
            color = "#7650a0" if kind == "moderation" else "#344e60" if kind == "direct" else "#87939d"
            cluster["edges"].append({"tail": start, "head": end, "attributes": {
                "label": _label(label), "color": color, "fontcolor": color,
                "style": "solid" if kind in {"direct", "moderation"} else "dashed",
                "arrowhead": "normal" if arrow else "none"}})
            manifest["edges"].append({"from": start, "to": end, "relation_id": rid,
                                      "kind": kind, "label": label, "arrow": arrow})

        direct_counts = Counter((r["cause"], r["effect"]) for _, r, _ in records if _effect_type(r) == "direct")
        junctions = {}
        segments = {}
        for rid, relation, mediators in records:
            kind = _effect_type(relation)
            if kind == "moderation":
                segment = (relation["cause"], relation["effect"])
            elif kind == "indirect" and relation.get("moderator") and relation.get("moderated_path"):
                path = [relation["cause"], *mediators, relation["effect"]]
                segment = tuple(relation["moderated_path"])
                if segment not in list(zip(path, path[1:])):
                    raise ValueError("moderated_path must be an explicitly identified segment of the mediator chain")
            else:
                continue
            segments[rid] = segment
            if segment not in junctions:
                junctions[segment] = node("", "junction", segment)

        # A unique direct effect can share its path junction with a moderator.
        # Conflicting direct estimates stay separate, with a dashed moderation scaffold.
        drawn = set()

        def path_edge(a, b, rid, kind, label="", use_junction=True):
            pair = (a, b)
            junction = junctions.get(pair) if use_junction else None
            if junction:
                edge(node(a), junction, rid, kind, arrow=False)
                edge(junction, node(b), rid, kind, label)
            else:
                edge(node(a), node(b), rid, kind, label)
            drawn.add(pair)

        for rid, relation, _ in records:
            if _effect_type(relation) == "direct":
                pair = (relation["cause"], relation["effect"])
                path_edge(*pair, rid, "direct", _statistics(relation), direct_counts[pair] == 1)

        scaffolded = set()
        for rid, relation, mediators in records:
            kind = _effect_type(relation)
            if kind == "direct":
                continue
            path = [relation["cause"], *mediators, relation["effect"]]
            # Unknown mediation is displayed in a note, not as a purported direct edge.
            if kind == "moderation" or mediators:
                for pair in zip(path, path[1:]):
                    needs_junction = pair in junctions and direct_counts[pair] != 1
                    if (pair not in drawn or needs_junction) and pair not in scaffolded:
                        path_edge(*pair, rid, "context")
                        scaffolded.add(pair)
            note = None
            if kind == "indirect":
                description = " → ".join(path) if mediators else f"{path[0]} → [mediator unreported] → {path[-1]}"
                label = f"Indirect: {description}\n{_statistics(relation)}"
                if relation.get("moderator"):
                    label += f"\nModerator: {relation['moderator']}"
                    if rid not in segments:
                        label += "\nModerated segment unspecified"
                        manifest["warnings"].append({"relation_id": rid, "kind": "unknown_moderated_segment"})
                note = node(label, "note", rid)
                for name in path:
                    node(name)
            if kind == "moderation" or relation.get("moderator"):
                moderator = relation.get("moderator")
                mid = node(moderator) if moderator else node(f"Moderator unreported ({rid})", "unknown", rid)
                if rid in segments:
                    label = _statistics(relation) if kind == "moderation" else ""
                    edge(mid, junctions[segments[rid]], rid, "moderation", label)
                else:
                    edge(mid, note, rid, "context", arrow=False)
        graph["subgraphs"].append(cluster)
    if not relations:
        graph["nodes"].append({"name": "empty", "attributes": {
            "shape": "plaintext", "label": "No extracted relations — inspect original for omissions"}})
    return graph, manifest


def render_graph(relations, output, relation_ids=None, direction="LR", title="Extracted SEM relations", engine="dot"):
    """Render SVG/PNG through Python graphviz and the system dot executable."""
    output = Path(output).resolve()
    if output.suffix not in {".svg", ".png"}:
        raise ValueError("Graph output must end in .svg or .png")
    try:
        import graphviz
    except ImportError as error:
        raise RuntimeError("Graph rendering requires the Python graphviz package: pip install graphviz") from error

    graph, manifest = build_graph(relations, relation_ids, direction, title)
    if engine not in {"dot", "neato", "fdp"}:
        raise ValueError("Use a Graphviz engine: dot, neato, or fdp")
    dot = graphviz.Digraph(engine=engine)

    def attributes(values):
        # _label already protects literal backslashes. Preserve line breaks and
        # quote <...> labels as ordinary text rather than Graphviz HTML markup.
        return {key: graphviz.nohtml((str(value).lower() if isinstance(value, bool) else str(value))
                                    .replace("\n", r"\n"))
                for key, value in values.items()}

    def populate(target, data):
        target.attr("graph", **attributes(data.get("graphAttributes", {})))
        target.attr("node", **attributes(data.get("nodeAttributes", {})))
        target.attr("edge", **attributes(data.get("edgeAttributes", {})))
        for node in data.get("nodes", []):
            target.node(node["name"], **attributes(node["attributes"]))
        for edge in data.get("edges", []):
            target.edge(edge["tail"], edge["head"], **attributes(edge["attributes"]))
        for cluster in data.get("subgraphs", []):
            with target.subgraph(name=cluster["name"]) as subgraph:
                populate(subgraph, cluster)

    populate(dot, graph)
    try:
        svg = dot.pipe(format="svg")
        rendered = dot.pipe(format="png") if output.suffix == ".png" else svg
    except graphviz.ExecutableNotFound as error:
        raise RuntimeError("Install system Graphviz and put dot on PATH "
                           "(macOS: brew install graphviz; Linux/Colab: apt-get install graphviz)") from error
    except graphviz.CalledProcessError as error:
        detail = (error.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Graphviz rendering failed: {detail}") from error

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rendered)
    output.with_suffix(".dot").write_text(dot.source, encoding="utf-8")
    if output.suffix == ".png":
        output.with_suffix(".svg").write_bytes(svg)
    manifest["image"] = str(output)
    write_json(output.with_suffix(".graph.json"), manifest)
    return manifest
