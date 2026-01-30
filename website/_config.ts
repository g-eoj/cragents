import lume from "lume/mod.ts";
import codeHighlight from "lume/plugins/code_highlight.ts";

const site = lume({
  src: ".",
  dest: "../docs",
});

site.use(codeHighlight());
site.copy("styles.css");
site.copy("main.js");
site.copy("favicon.ico");

export default site;
