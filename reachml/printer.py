import pandas as pd

# todo: control indices of points to print
# todo: flag for whether to print actions


class ReachableSetPrinter(object):
    """
    Working class to print the points in a ReachableSets in a table that can be
    included in a publication or website. This can eventually be appended to ReachableSet
    """

    def __init__(self, reachable_set):
        assert isinstance(reachable_set)
        self.reachable_set = reachable_set
        raise NotImplementedError()

    def to_flat_df(self):
        """converts points to a data.frame"""
        raise NotImplementedError()
        tex_columns = ["features", "x", "x_new"]
        tex_df = self._df[tex_columns]
        if len(tex_df) == 0:
            return []

        # split components for each item
        tex_df = tex_df.reset_index().rename(columns={"index": "item_id"})
        df_list = []
        for n in tex_columns:
            tmp = tex_df.set_index(["item_id"])[n].apply(pd.Series).stack()
            tmp = tmp.reset_index().rename(columns={"level_1": "var_id"})
            tmp_name = tmp.columns[-1]
            tmp = tmp.rename(columns={tmp_name: n})
            df_list.append(tmp)

        # combine into a flattened list
        flat_df = df_list[0]
        for k in range(1, len(df_list)):
            flat_df = flat_df.merge(df_list[k])

        # drop the merge index
        flat_df = flat_df.drop(columns=["var_id"])

        # index items by item_id
        flat_df = flat_df.sort_values(by="item_id")
        flat_df = flat_df.rename(
            columns={
                "item_id": "item",
                "features": "Features to Change",
                "x": "Current Value",
                "x_new": "Required Value",
            }
        )
        return flat_df.set_index("item")

    def to_latex(self, name_formatter="\\textit"):
        """
        converts current Flipset to Latex table
        :param name_formatter:
        :return:
        """
        raise NotImplementedError()
        flat_df = self.to_flat_df()

        # add another column for the latex arrow symbol
        idx = flat_df.columns.tolist().index("Required Value")
        flat_df.insert(loc=idx, column="to", value=["longrightarrow"] * len(flat_df))

        # name headers
        flat_df = flat_df.rename(
            columns={
                "features": "\textsc{Feature Subset}",
                "Current Value": "\textsc{Current Values}",
                "Required Value": "\textsc{Required Values}",
            }
        )

        # get raw tex table
        table = flat_df.to_latex(
            multirow=True, index=True, escape=False, na_rep="-", column_format="rlccc"
        )

        # manually wrap names with a formatter function
        if name_formatter is not None:
            for v in self._variable_names:
                table = table.replace(v, "%s{%s}" % (name_formatter, v))

        # add the backslash for the arrow
        table = table.replace("longrightarrow", "$\\longrightarrow$")

        # minor embellishments
        table = table.split("\n")
        table[2] = table[2].replace("to", "")
        table[2] = table[2].replace("{}", "")
        table.pop(3)
        table.pop(3)
        out = "\n".join(table)
        return out
