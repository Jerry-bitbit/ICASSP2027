"""Check physical figure sizes and ordinary text before exporting vector PDF."""
import json
from itertools import combinations
import matplotlib.pyplot as plt


def visible_text(fig):
    texts = list(fig.texts)
    for ax in fig.axes:
        texts.extend(ax.texts)
        if ax.axison:
            texts += [ax.title, ax.xaxis.label, ax.yaxis.label]
            texts += ax.get_xticklabels() + ax.get_yticklabels()
        if ax.get_legend() is not None:
            texts += ax.get_legend().get_texts()
    for legend in fig.legends:
        texts += legend.get_texts()
    return [t for t in texts if t.get_visible() and t.get_text()]


def save_figure(fig, root, stem, minimum, text_objects=None):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = visible_text(fig) if text_objects is None else text_objects
    texts = [t for t in texts if t.get_visible() and t.get_text()]
    records = []
    outside = []
    boxes = []
    for t in texts:
        rect = t.get_window_extent(renderer)
        boxes.append(rect)
        records.append(dict(text=t.get_text(), fontsize_pt=t.get_fontsize(),
                            bbox_pixels=list(rect.bounds)))
        if (rect.x0 < -.5 or rect.y0 < -.5 or
                rect.x1 > fig.bbox.width + .5 or rect.y1 > fig.bbox.height + .5):
            outside.append(t.get_text())
    overlaps = [(texts[i].get_text(), texts[j].get_text())
                for i, j in combinations(range(len(texts)), 2)
                if boxes[i].overlaps(boxes[j])]
    result = dict(figure=stem, width_mm=float(fig.get_figwidth()*25.4),
                  height_mm=float(fig.get_figheight()*25.4),
                  min_base_font_pt=min(t.get_fontsize() for t in texts),
                  text_outside=outside, overlapping_text=overlaps, texts=records,
                  note="Base text sizes; mathematical subscripts/superscripts retain normal typesetting.")
    (root/'verification').mkdir(exist_ok=True)
    (root/'figures').mkdir(exist_ok=True)
    (root/'verification'/f'{stem}_layout.json').write_text(json.dumps(result,indent=2)+'\n')
    assert result['min_base_font_pt'] >= minimum, result
    assert not outside, f'{stem}: text outside figure: {outside}'
    # Do not use bbox_inches='tight': preserve the declared physical width.
    fig.savefig(root/'figures'/f'{stem}.pdf', facecolor='white')
    fig.savefig(root/'figures'/f'{stem}.png', dpi=300, facecolor='white')
    print(stem, 'base font', result['min_base_font_pt'], 'pt; text overlaps', overlaps)
